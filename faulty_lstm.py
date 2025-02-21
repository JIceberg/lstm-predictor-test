import torch
import torch.nn as nn
import torch.nn.functional as F
import struct
import random
import math
import numpy as np

np.random.seed(69)

class FaultyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, error_rate=0, k=3):
        super(FaultyLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers    

        self.cells = nn.ModuleList([nn.LSTMCell(input_size if i == 0 else hidden_size, hidden_size)
                                  for i in range(num_layers)])

        self.error_rate = error_rate
        self.k = k
        self.mean_grad = {}
        self.var_grad = {}
        self.num_updates = {}

    def forward(self, x, hidden=None, compute_grad=False, inject_faults=False):
        batch_size, seq_len, _ = x.shape

        if hidden is None:
            h_t = [torch.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
            c_t = [torch.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
        else:
            h_t, c_t = hidden

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]    # input at timestep t
            
            for i, cell in enumerate(self.cells):
                h_t[i], c_t[i] = cell(x_t, (h_t[i], c_t[i]))
                if not self.training:   # inject only during inference
                    if not compute_grad and inject_faults:   # compute gradient over clean data only
                        h_t[i] = self.bit_flip_fault_inj(h_t[i])
                        c_t[i] = self.bit_flip_fault_inj(c_t[i])
                    
                    if compute_grad:
                        self.update_running_statistics(h_t[i], f"h[{i}]")
                        self.update_running_statistics(c_t[i], f"c[{i}]")
        
                    if not compute_grad and self.mean_grad:   # only threshold if gradients are already calculated
                        h_t[i] = self.threshold_gradients(h_t[i], f"h[{i}]")
                        c_t[i] = self.threshold_gradients(c_t[i], f"c[{i}]")
                x_t = h_t[i]
            
            outputs.append(h_t[-1].unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        outputs = outputs.transpose(0, 1).contiguous()
        return outputs, (h_t, c_t)
    
    def bit_flip_fault_inj(self, output):
        # Flatten tensor for easier manipulation
        flat_output = output.view(-1)

        # Convert float tensor to int representation (IEEE 754)
        float_bits = flat_output.to(torch.float32).cpu().numpy().view(np.uint32)

        # Randomly select bits to flip
        num_elements = flat_output.numel()
        random_bits = np.random.randint(0, 32, size=num_elements, dtype=np.uint32)

        # Create a mask to determine which values to flip
        flip_mask = np.random.rand(num_elements) < self.error_rate

        # Perform bitwise XOR only for selected neurons
        flipped_bits = float_bits ^ (1 << random_bits)
        
        # Ensure numerical stability (avoid NaN, Inf)
        flipped_vals = flipped_bits.view(np.float32)
        valid_mask = np.isfinite(flipped_vals)
        flipped_vals[~valid_mask] = flat_output.cpu().numpy()[~valid_mask]  # Restore original if NaN/Inf

        # Replace only values where flip_mask is True
        float_bits[flip_mask] = flipped_bits[flip_mask]

        # Convert back to PyTorch tensor
        modified_output = torch.tensor(float_bits.view(np.float32), dtype=torch.float32, device=output.device).view(output.shape)

        return modified_output

    def update_running_statistics(self, layer, layer_name):
        layer_unsqeueezed = layer.unsqueeze(1)
        kernel = torch.tensor([-1.0, 1.0, 0.0]).view(1, 1, 3)
        grad_Y = F.conv1d(layer_unsqeueezed, kernel.to(layer.device), padding=1).squeeze(1)

        mean = grad_Y.mean(dim=0).cpu().detach().numpy()
        var = grad_Y.var(dim=0).cpu().detach().numpy()

        if layer_name not in self.mean_grad:
            self.mean_grad[layer_name] = mean
            self.var_grad[layer_name] = var
            self.num_updates[layer_name] = 0
        else:
            self.num_updates[layer_name] += 1
            self.mean_grad[layer_name] += (mean - self.mean_grad[layer_name]) / self.num_updates[layer_name]
            self.var_grad[layer_name] += (var - self.var_grad[layer_name]) / self.num_updates[layer_name]
    
    def threshold_gradients(self, layer, layer_name):
        if self.mean_grad[layer_name] is None or self.var_grad[layer_name] is None:
            return layer
        
        std_grad = np.sqrt(self.var_grad[layer_name])

        mean_grad_tensor = torch.tensor(self.mean_grad[layer_name], device=layer.device)
        std_grad_tensor = torch.tensor(std_grad, device=layer.device)

        lower_bound = mean_grad_tensor - self.k * std_grad_tensor
        upper_bound = mean_grad_tensor + self.k * std_grad_tensor

        layer_unsqeueezed = layer.unsqueeze(1)
        kernel = torch.tensor([-1.0, 1.0, 0.0]).view(1, 1, 3)
        grad_Y = F.conv1d(layer_unsqeueezed, kernel.to(layer.device), padding=1).squeeze(1)

        mask = (grad_Y < lower_bound) | (grad_Y > upper_bound)
        masked_layer = layer.clone()
        masked_layer[mask] = 0

        return masked_layer