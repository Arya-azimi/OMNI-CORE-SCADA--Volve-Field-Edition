import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import joblib
import warnings

warnings.filterwarnings('ignore')

# --- SECTION 1: PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Volve production data.xlsx')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

print("==================================================")
print(" MAGNORA // TIME-SERIES FORECASTER (PYTORCH LSTM)")
print("==================================================")

# --- SECTION 2: DATA EXTRACTION & SEQUENCING ---
print("[SYSTEM] Loading operational data for Time-Series analysis...")
df = pd.read_excel(DATA_PATH, sheet_name='Daily Production Data')

# Target a highly productive well (15/9-F-14 is a primary oil producer in Volve)
target_well = '15/9-F-14'
df_well = df[(df['NPD_WELL_BORE_NAME'] == target_well) & (df['FLOW_KIND'] == 'production')]
df_well = df_well.sort_values('DATEPRD').dropna(subset=['BORE_OIL_VOL'])
df_well = df_well[df_well['BORE_OIL_VOL'] > 0]

data = df_well[['BORE_OIL_VOL']].values

print(f"[INFO] Time-Series points acquired for well {target_well}: {len(data)}")

if len(data) == 0:
    print(f"[FATAL] No valid production data found for {target_well}. Check well identifier.")
    exit()

# Scale the data for Neural Network stability
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create sequences (Look back 30 days to predict the next day)
SEQ_LENGTH = 30


def create_sequences(dataset, seq_length):
    xs, ys = [], []
    for i in range(len(dataset) - seq_length):
        x = dataset[i:(i + seq_length)]
        y = dataset[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


X, y = create_sequences(scaled_data, SEQ_LENGTH)

X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

train_size = int(len(X) * 0.8)
X_train, y_train = X_tensor[:train_size], y_tensor[:train_size]
X_test, y_test = X_tensor[train_size:], y_tensor[train_size:]


# --- SECTION 3: PYTORCH LSTM ARCHITECTURE ---
class DeclineCurveLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(DeclineCurveLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


model = DeclineCurveLSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# --- SECTION 4: DEEP LEARNING TRAINING LOOP ---
print("[SYSTEM] Initializing PyTorch LSTM training sequence...")
EPOCHS = 150

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()

    predictions = model(X_train)
    loss = criterion(predictions, y_train)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 30 == 0:
        model.eval()
        with torch.no_grad():
            val_preds = model(X_test)
            val_loss = criterion(val_preds, y_test)
        print(f"Epoch [{epoch + 1}/{EPOCHS}] | Train Loss: {loss.item():.6f} | Val Loss: {val_loss.item():.6f}")

# --- SECTION 5: MODEL EXPORT ---
model_path = os.path.join(MODELS_DIR, 'lstm_forecaster.pth')
scaler_path = os.path.join(MODELS_DIR, 'forecaster_scaler.pkl')

torch.save(model.state_dict(), model_path)
joblib.dump(scaler, scaler_path)

print(f"\n[SUCCESS] PyTorch Weights exported to: {model_path}")
print(f"[SUCCESS] LSTM Scaler exported to: {scaler_path}")
print("==================================================")