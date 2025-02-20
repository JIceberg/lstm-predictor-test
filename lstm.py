import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import struct
import random
import os
import math
import sys

error_rate = 0
if len(sys.argv) > 1:
    error_rate = float(sys.argv[1])

df = pd.read_csv("data/WTH.csv")

df['date'] = pd.to_datetime(df['date'])
df['hour'] = df['date'].dt.hour
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

df.drop(['date'], axis=1, inplace=True)

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

SEQ_LENGTH = 24  # Using past 24 hours to predict next hour

def create_sequences(data, seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

X, y = create_sequences(df_scaled.values, SEQ_LENGTH)

X_train, y_train = torch.tensor(X[:-100], dtype=torch.float32), torch.tensor(y[:-100], dtype=torch.float32)
X_test, y_test = torch.tensor(X[-100:], dtype=torch.float32), torch.tensor(y[-100:], dtype=torch.float32)

BATCH_SIZE = 32
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

class WeatherLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, error_rate=0, k=3):
        super(WeatherLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.cells = nn.ModuleList([nn.LSTMCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        self.fc = nn.Linear(hidden_size, output_size)

        self.error_rate = error_rate
        self.k = k
        self.mean_grad = {}
        self.var_grad = {}
        self.num_updates = {}

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h_t = torch.zeros(batch_size, self.hidden_size)
        c_t = torch.zeros(batch_size, self.hidden_size)

        for t in range(seq_len):
            x_t = x[:, t, :]    # input at timestep t
            for layer in self.cells:
                h_t, c_t = layer(x_t, (h_t, c_t))
                if self.training:
                    self.update_running_statistics(h_t, f"h_{t}")
                    self.update_running_statistics(c_t, f"c_{t}")
                if not self.training:   # inject only during inference
                    h_t, c_t = self.bit_flip_fault_inj(h_t), self.bit_flip_fault_inj(c_t)
                    h_t = self.threshold_gradients(h_t, f"h_{t}")
                    c_t = self.threshold_gradients(c_t, f"c_{t}")
                x_t = h_t
        
        output = self.fc(h_t)
        # if not self.training:
        #     output = self.bit_flip_fault_inj(output)
        return output
    
    def bit_flip_fault_inj(self, output):
        flat_output = output.flatten()
        
        # flip a random bit in the value
        for neuron in range(flat_output.shape[0]):
            float_val = flat_output[neuron]
            float_bits = struct.unpack('>I', struct.pack('>f', float_val))[0]
            random_bit = random.randint(0, 31)
            flipped_bits = float_bits ^ (1 << random_bit)
            flipped_val = struct.unpack('>f', struct.pack('>I', flipped_bits))[0]

            if math.isnan(flipped_val) or math.isinf(flipped_val):
                flipped_val = float_val

            # set neuron output to flipped value
            if random.random() < self.error_rate:
                print("injecting error")
                flat_output[neuron] = flipped_val

        modified_output = flat_output.reshape(output.shape)
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

INPUT_DIM = X_train.shape[2]
HIDDEN_DIM = 64
NUM_LAYERS = 2
OUTPUT_DIM = y_train.shape[1]

model = WeatherLSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM, error_rate=error_rate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# model.load_state_dict(torch.load("saved_model.pth", weights_only=False))

EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss/len(train_loader):.4f}")

model.eval()
with torch.no_grad():
    predictions, actuals = [], []
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        pred = model(batch_X)
        predictions.append(pred.cpu().numpy())
        actuals.append(batch_y.cpu().numpy())

predictions = np.concatenate(predictions, axis=0)
actuals = np.concatenate(actuals, axis=0)

plt.figure(figsize=(12, 6))
plt.plot(actuals[:, 1], label="Actual Temperature (Scaled)", color='blue')
plt.plot(predictions[:, 1], label="Predicted Temperature (Scaled)", linestyle="dashed", color='red')
plt.legend()
plt.title("Weather Prediction with LSTM")
plt.show()
