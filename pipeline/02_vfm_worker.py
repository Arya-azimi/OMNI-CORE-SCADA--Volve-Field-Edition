import os
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# --- SECTION 1: PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Volve production data.xlsx')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(MODELS_DIR, exist_ok=True)

print("==================================================")
print(" MAGNORA // VIRTUAL FLOW METERING (VFM) WORKER")
print("==================================================")

# --- SECTION 2: DATA EXTRACTION & FILTERING ---
print("[SYSTEM] Extracting Volve operational dataset...")
try:
    df = pd.read_excel(DATA_PATH, sheet_name='Daily Production Data')
except Exception as e:
    print(f"[FATAL ERROR] Dataset ingestion failed: {e}")
    exit()

print("[SYSTEM] Filtering non-producing wells and cleaning noise...")
# Isolate active production wells only
df_prod = df[df['FLOW_KIND'] == 'production'].copy()

# Define core features (X) and target variable (y)
FEATURES = ['AVG_WHP_P', 'AVG_WHT_P', 'AVG_DP_TUBING', 'AVG_CHOKE_SIZE_P']
TARGET = 'BORE_OIL_VOL'

# Drop missing telemetry and zero-production days
df_clean = df_prod.dropna(subset=FEATURES + [TARGET])
df_clean = df_clean[df_clean[TARGET] > 0]

X = df_clean[FEATURES]
y = df_clean[TARGET]

print(f"[INFO] Valid telemetry vectors acquired: {len(X)}")

# --- SECTION 3: NEURAL SCALING & SPLITTING ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.15, random_state=42
)

# --- SECTION 4: XGBOOST ALGORITHM TRAINING ---
print("[SYSTEM] Initializing XGBoost Regressor engine...")
vfm_model = xgb.XGBRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    tree_method='hist',
    early_stopping_rounds=50,
    random_state=42
)

print("[SYSTEM] Commencing AI training sequence. Awaiting convergence...")
vfm_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)

# --- SECTION 5: MODEL EVALUATION & EXPORT ---
y_pred = vfm_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- VFM PERFORMANCE METRICS ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} bbl/day")
print(f"R2 Variance Score:         {r2:.4f}")

# Extract feature importance
print("\n--- SENSOR IMPORTANCE ---")
importances = vfm_model.feature_importances_
for sensor, imp in zip(FEATURES, importances):
    print(f"{sensor}: {imp*100:.2f}%")

model_file = os.path.join(MODELS_DIR, 'vfm_regressor.json')
scaler_file = os.path.join(MODELS_DIR, 'vfm_scaler.pkl')

vfm_model.save_model(model_file)
joblib.dump(scaler, scaler_file)

print(f"\n[SUCCESS] AI Weights exported to: {model_file}")
print(f"[SUCCESS] Scaler exported to: {scaler_file}")
print("==================================================")