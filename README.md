# 🌐 MAGNORA | OMNI-CORE SCADA (Volve Field Edition)

## ⚡ Executive Summary

The Magnora OMNI-CORE SCADA is a production-grade, multi-engine Artificial Intelligence platform designed to monitor, forecast, and optimize upstream oil and gas operations. Built on top of the historical Equinor Volve Field dataset, this system replaces traditional physics-based simulators with ultra-fast, data-driven machine learning models.

The platform features a Single Page Application (SPA) dashboard powered by FastAPI and WebSockets, enabling real-time, low-latency streaming of telemetry data directly into the AI engines for instant inference.

## 🧠 The 4 AI Engines

### 1. Virtual Flow Metering (VFM)

- Architecture: XGBRegressor

- Purpose: Eliminates the need for expensive physical multi-phase flow meters. It predicts real-time daily oil production (BORE_OIL_VOL) by analyzing wellhead pressure, temperature, tubing dynamics, and choke size.

### 2. Production Forecasting (Decline Curve Analysis)

- Architecture: PyTorch LSTM (Long Short-Term Memory)

- Purpose: Analyzes 30-day sequential lookback windows of production history to understand the field's natural depletion rate and accurately forecast future output.

### 3. Predictive Maintenance (Anomaly Detection)

- Architecture: Isolation Forest (Unsupervised Learning)

- Purpose: Continuously scans downhole pressure and temperature arrays. It flags systemic deviations and hardware stress events (anomalies) in real-time, preventing catastrophic equipment failures before they occur.

### 4. Waterflood Optimizer

- Architecture: XGBRegressor (Physics-Informed with L1/L2 Regularization)

- Purpose: Models the thermodynamic "sponge effect" of the reservoir. By utilizing moving averages (7-day/30-day rolling windows) of water injection, it predicts the delayed oil recovery response, optimizing field pressure maintenance.

## 🏗️ System Architecture & Directory Tree
```
volve-omni-production/
├── app/
│   └── app_fastapi.py          # Core ASGI Server & WebSocket Streamer
├── pipeline/
│   ├── 01_data_inspector.py    # Ingestion & schema validation protocol
│   ├── 02_vfm_worker.py        # XGBoost VFM training script
│   ├── 03_forecaster.py        # PyTorch LSTM training script
│   ├── 04_anomaly_engine.py    # Isolation Forest training script
│   └── 05_optimizer.py         # Waterflooding optimization training script
├── models/
│   ├── vfm_regressor.json      # Pre-trained VFM weights
│   ├── lstm_forecaster.pth     # Pre-trained Deep Learning weights
│   ├── anomaly_detector.pkl    # Serialized Isolation Forest
│   ├── waterflood_optimizer.json # Pre-trained Waterflood weights
│   └── *_scaler.pkl            # State standardizers for all modules
├── data/
│   └── Volve production data.xlsx # Raw operational dataset
├── README.md                   # System documentation
└── requirements.txt            # Dependency manifest
```

## 🚀 Getting Started

### Prerequisites

- Python 3.12+

### 1. Installation

Clone the repository and install the high-performance computing dependencies:
```
git clone [https://github.com/arya-developer/volve-omni-production.git](https://github.com/arya-developer/volve-omni-production.git)
cd volve-omni-production
pip install -r requirements.txt
```

2. Launching the SCADA Dashboard

Start the asynchronous FastAPI server:
```
python3 app_fastapi.py
```

Navigate to http://127.0.0.1:8000 in your browser. Click "▶ START STREAM" to initiate the WebSocket telemetry feed.

Developed by Magnora — Architecting the Future of Industrial AI.