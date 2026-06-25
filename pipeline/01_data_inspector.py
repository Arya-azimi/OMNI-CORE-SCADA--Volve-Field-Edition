import pandas as pd
import numpy as np
import os

# --- Section 1: Path Configuration ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
FILE_NAME = "/home/arya/arya/DL_MODELS/18-volve-omni-production/data/Volve production data.xlsx"
FILE_PATH = os.path.join(BASE_DIR, FILE_NAME)

print("==================================================")
print(" MAGNORA // DATA INGESTION & VALIDATION PROTOCOL")
print("==================================================")

# --- Section 2: File Extraction ---
try:
    xl = pd.ExcelFile(FILE_PATH)
    print(f"[SUCCESS] Excel file loaded.")
    print(f"[INFO] Discovered Sheets: {xl.sheet_names}")

    target_sheet = 'Daily Production Data'
    if target_sheet in xl.sheet_names:
        df = xl.parse(target_sheet)
    else:
        df = xl.parse(0)

    # --- Section 3: Dimensionality & Schematics ---
    print("\n--- DATASET OVERVIEW ---")
    print(f"Total Rows: {df.shape[0]}")
    print(f"Total Columns: {df.shape[1]}")

    print("\n--- COLUMN SCHEMATICS & INTEGRITY ---")
    info_df = pd.DataFrame({
        'Data Type': df.dtypes,
        'Valid Entries': df.notnull().sum(),
        'Missing (%)': (df.isnull().sum() / len(df) * 100).round(2)
    })
    print(info_df.to_string())

    # --- Section 4: Well Identification ---
    print("\n--- WELL IDENTIFIERS ---")
    well_col_candidates = ['NPD_WELL_BORE_NAME', 'WELLBORE', 'Well']
    for col in well_col_candidates:
        if col in df.columns:
            wells = df[col].unique()
            print(f"Unique Wells Found ({len(wells)}): {wells}")
            break

    print("\n[SYSTEM] Validation protocol complete. Awaiting architecture decisions.")

except FileNotFoundError:
    print(f"[FATAL] File not found at path: {FILE_PATH}")
except Exception as e:
    print(f"[ERROR] Process terminated unexpectedly: {e}")