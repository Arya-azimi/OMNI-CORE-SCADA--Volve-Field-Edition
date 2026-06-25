import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# --- SECTION 1: PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Volve production data.xlsx')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

print("==================================================")
print(" MAGNORA // ANOMALY ENGINE (PREDICTIVE MAINTENANCE)")
print("==================================================")

# --- SECTION 2: DATA EXTRACTION & FILTERING ---
print("[SYSTEM] Ingesting telemetry data for Anomaly Detection...")
df = pd.read_excel(DATA_PATH, sheet_name='Daily Production Data')

# Isolate producing wells
df_prod = df[df['FLOW_KIND'] == 'production']

# Select critical physical sensors for hardware health monitoring
FEATURES = ['AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_WHP_P', 'AVG_WHT_P']

# Filter out missing downhole/wellhead sensor data
df_clean = df_prod.dropna(subset=FEATURES)
X = df_clean[FEATURES]

print(f"[INFO] Operational telemetry vectors acquired: {len(X)}")

if len(X) == 0:
    print("[FATAL] Insufficient sensor data for anomaly detection.")
    exit()

# --- SECTION 3: NEURAL SCALING ---
print("[SYSTEM] Scaling sensor arrays...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- SECTION 4: ISOLATION FOREST TRAINING ---
print("[SYSTEM] Initializing Isolation Forest (Unsupervised Learning)...")
# Contamination defines the expected percentage of anomalies in the dataset (e.g., 2%)
anomaly_model = IsolationForest(
    n_estimators=200,
    max_samples='auto',
    contamination=0.02,
    random_state=42,
    n_jobs=-1
)

print("[SYSTEM] Scanning for systemic deviations and hardware anomalies...")
anomaly_model.fit(X_scaled)

# Calculate anomaly distribution across the historical data
predictions = anomaly_model.predict(X_scaled)
anomaly_count = len(predictions[predictions == -1])
normal_count = len(predictions[predictions == 1])

print("\n--- HEALTH DIAGNOSTIC REPORT ---")
print(f"Normal Operations Detected: {normal_count} logs")
print(f"System Anomalies Detected:  {anomaly_count} logs")
print(f"Anomaly Ratio:              {(anomaly_count/len(predictions))*100:.2f}%")

# --- SECTION 5: MODEL EXPORT ---
model_path = os.path.join(MODELS_DIR, 'anomaly_detector.pkl')
scaler_path = os.path.join(MODELS_DIR, 'anomaly_scaler.pkl')

joblib.dump(anomaly_model, model_path)
joblib.dump(scaler, scaler_path)

print(f"\n[SUCCESS] Isolation Forest exported to: {model_path}")
print(f"[SUCCESS] Anomaly Scaler exported to: {scaler_path}")
print("==================================================")