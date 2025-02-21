import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import sys
import os
from faulty_lstm import FaultyLSTM

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

class WeatherPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, error_rate=0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.lstm = FaultyLSTM(input_dim, hidden_dim, num_layers=num_layers, error_rate=error_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x, hidden, compute_grad=False, inject_faults=False):
        batch_size = x.size(0)

        lstm_out, hidden = self.lstm(
            x,
            hidden=hidden,
            compute_grad=compute_grad,
            inject_faults=inject_faults
        )
        out = self.fc(lstm_out)
        # out = self.tanh(out)

        return out, hidden

SEQ_LENGTH = 24

def create_sequences(data, seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length * 2):
        seq = data[i:i + seq_length]
        label = data[i + seq_length:i + seq_length * 2]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

X, y = create_sequences(df_scaled.values, SEQ_LENGTH)

X_train, y_train = torch.tensor(X[:-100], dtype=torch.float32), torch.tensor(y[:-100], dtype=torch.float32)
X_test, y_test = torch.tensor(X[-100:], dtype=torch.float32), torch.tensor(y[-100:], dtype=torch.float32)

BATCH_SIZE = 32
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

INPUT_DIM = X_train.shape[2]
HIDDEN_DIM = 64
OUTPUT_DIM = y_train.shape[2]

model = WeatherPredictor(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, num_layers=2, error_rate=error_rate)
print(model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, data, epochs=10):
    model.train()
    for epoch in range(epochs):
        h = None
        with tqdm(data, unit="batch") as tepoch:
            for batch_X, batch_y in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")

                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                if h != None:
                    h = tuple([[data.detach() for data in each] for each in h])

                optimizer.zero_grad()
                outputs, h = model(batch_X, h)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss = loss.item()
                tepoch.set_postfix(loss=epoch_loss)

if os.path.exists('saved_model.pth'):
    model.load_state_dict(torch.load("saved_model.pth", weights_only=True))
else:
    train_model(model, train_loader, epochs=10)
    torch.save(model.state_dict(), "saved_model.pth")
    
def evaluate_model(model, data, compute_grad=False, inject_faults=False, plot=False):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        h = None
        for batch_X, batch_y in data:
            if h != None:
                h = tuple([[data.detach() for data in each] for each in h])
            pred, h = model(batch_X, h, compute_grad=compute_grad, inject_faults=inject_faults)
            pred = torch.clamp(pred, min=0.0, max=1.0)
            predictions.append(pred.cpu().numpy())
            actuals.append(batch_y.cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(actuals[32, :, 1], label="Actual Temperature (Scaled)", color='blue')
        plt.plot(predictions[32, :, 1], label="Predicted Temperature (Scaled)", linestyle="dashed", color='red')
        plt.legend()
        plt.title("Weather Prediction with LSTM")
        plt.show()

# compute grads first
evaluate_model(model, train_loader, compute_grad=True)
print(model.lstm.mean_grad)

evaluate_model(model, test_loader, inject_faults=True, plot=True)