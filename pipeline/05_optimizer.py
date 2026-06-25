import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# --- SECTION 1: PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Volve production data.xlsx')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

print("==================================================")
print(" MAGNORA // WATERFLOODING OPTIMIZATION ENGINE")
print("==================================================")

# --- SECTION 2: FIELD-LEVEL AGGREGATION ---
print("[SYSTEM] Aggregating field-level injection and production telemetry...")
df = pd.read_excel(DATA_PATH, sheet_name='Daily Production Data')

injectors = df[df['FLOW_KIND'] == 'injection'].groupby('DATEPRD')['BORE_WI_VOL'].sum().reset_index()
injectors.rename(columns={'BORE_WI_VOL': 'TOTAL_WATER_INJECTED'}, inplace=True)

producers = df[df['FLOW_KIND'] == 'production'].groupby('DATEPRD')['BORE_OIL_VOL'].sum().reset_index()
producers.rename(columns={'BORE_OIL_VOL': 'TOTAL_OIL_PRODUCED'}, inplace=True)

field_df = pd.merge(injectors, producers, on='DATEPRD', how='inner')
field_df = field_df[(field_df['TOTAL_WATER_INJECTED'] > 0) & (field_df['TOTAL_OIL_PRODUCED'] > 0)]

# --- SECTION 3: ADVANCED RESERVOIR PHYSICS (ROLLING WINDOWS) ---
print("[SYSTEM] Computing thermodynamic pressure build-up (Rolling Windows)...")
# A reservoir acts like a sponge. We calculate rolling means to represent pressure build-up.
field_df['ROLLING_INJ_7D'] = field_df['TOTAL_WATER_INJECTED'].rolling(window=7).mean()
field_df['ROLLING_INJ_30D'] = field_df['TOTAL_WATER_INJECTED'].rolling(window=30).mean()
field_df['CUMULATIVE_INJ'] = field_df['TOTAL_WATER_INJECTED'].cumsum() # Total energy added to reservoir

field_df['OIL_LAG_1'] = field_df['TOTAL_OIL_PRODUCED'].shift(1)
field_df['ROLLING_OIL_7D'] = field_df['TOTAL_OIL_PRODUCED'].rolling(window=7).mean()
field_df['DAYS_ONLINE'] = np.arange(len(field_df))

field_df.dropna(inplace=True)

FEATURES = ['TOTAL_WATER_INJECTED', 'ROLLING_INJ_7D', 'ROLLING_INJ_30D', 'CUMULATIVE_INJ', 'DAYS_ONLINE', 'OIL_LAG_1', 'ROLLING_OIL_7D']
TARGET = 'TOTAL_OIL_PRODUCED'

X = field_df[FEATURES]
y = field_df[TARGET]

# --- SECTION 4: SCALING & TRAINING ---
print("[SYSTEM] Initializing Field Optimizer AI (XGBoost)...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.15, random_state=42, shuffle=False
)

optimizer_model = xgb.XGBRegressor(
    n_estimators=1500,
    learning_rate=0.01,
    max_depth=4,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.8,      # L1 Regularization: Penalizes relying on just one feature (like OIL_LAG)
    reg_lambda=2.0,     # L2 Regularization: Smooths out the learning weights
    objective='reg:squarederror',
    random_state=42
)

optimizer_model.fit(X_train, y_train, verbose=False)

# --- SECTION 5: EVALUATION & EXPORT ---
y_pred = optimizer_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- WATERFLOOD OPTIMIZATION METRICS ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} bbl/day")
print(f"R2 Variance Score:         {r2:.4f}")

print("\n--- RESERVOIR RESPONSE TIME (Feature Importance) ---")
importances = optimizer_model.feature_importances_
for feature, imp in zip(FEATURES, importances):
    print(f"{feature}: {imp*100:.2f}%")

model_path = os.path.join(MODELS_DIR, 'waterflood_optimizer.json')
scaler_path = os.path.join(MODELS_DIR, 'waterflood_scaler.pkl')
optimizer_model.save_model(model_path)
joblib.dump(scaler, scaler_path)

print(f"\n[SUCCESS] Optimizer exported successfully.")
print("==================================================")