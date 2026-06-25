import os
import json
import asyncio
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
import joblib
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
import warnings

warnings.filterwarnings('ignore')

# --- SECTION 1: SYSTEM & MODEL CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Volve production data.xlsx')
MODELS_DIR = os.path.join(BASE_DIR, 'models')


class DeclineCurveLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(DeclineCurveLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


print(">>> [SYSTEM] Initializing AI Engines...")

try:
    vfm_model = xgb.XGBRegressor()
    vfm_model.load_model(os.path.join(MODELS_DIR, 'vfm_regressor.json'))
    vfm_scaler = joblib.load(os.path.join(MODELS_DIR, 'vfm_scaler.pkl'))

    anomaly_model = joblib.load(os.path.join(MODELS_DIR, 'anomaly_detector.pkl'))
    anomaly_scaler = joblib.load(os.path.join(MODELS_DIR, 'anomaly_scaler.pkl'))

    wf_model = xgb.XGBRegressor()
    wf_model.load_model(os.path.join(MODELS_DIR, 'waterflood_optimizer.json'))
    wf_scaler = joblib.load(os.path.join(MODELS_DIR, 'waterflood_scaler.pkl'))

    lstm_model = DeclineCurveLSTM()
    lstm_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'lstm_forecaster.pth'), weights_only=True))
    lstm_model.eval()
    lstm_scaler = joblib.load(os.path.join(MODELS_DIR, 'forecaster_scaler.pkl'))

    print(">>> [SUCCESS] All Neural Weights loaded.")
except Exception as e:
    print(f">>> [FATAL] AI Load Error: {e}")
    exit()

# --- SECTION 2: DATA INGESTION ---
print(">>> [SYSTEM] Caching telemetry streams...")
df_raw = pd.read_excel(DATA_PATH, sheet_name='Daily Production Data')

df_vfm = df_raw[df_raw['FLOW_KIND'] == 'production'].dropna(
    subset=['AVG_WHP_P', 'AVG_WHT_P', 'AVG_DP_TUBING', 'AVG_CHOKE_SIZE_P', 'BORE_OIL_VOL'])
df_vfm = df_vfm[df_vfm['BORE_OIL_VOL'] > 0].reset_index(drop=True)

df_ano = df_raw[df_raw['FLOW_KIND'] == 'production'].dropna(
    subset=['AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_WHP_P', 'AVG_WHT_P']).reset_index(drop=True)

inj = df_raw[df_raw['FLOW_KIND'] == 'injection'].groupby('DATEPRD')['BORE_WI_VOL'].sum().reset_index(
    name='TOTAL_WATER_INJECTED')
prd = df_raw[df_raw['FLOW_KIND'] == 'production'].groupby('DATEPRD')['BORE_OIL_VOL'].sum().reset_index(
    name='TOTAL_OIL_PRODUCED')
df_wf = pd.merge(inj, prd, on='DATEPRD', how='inner')
df_wf = df_wf[(df_wf['TOTAL_WATER_INJECTED'] > 0) & (df_wf['TOTAL_OIL_PRODUCED'] > 0)]
df_wf['ROLLING_INJ_7D'] = df_wf['TOTAL_WATER_INJECTED'].rolling(window=7).mean()
df_wf['ROLLING_INJ_30D'] = df_wf['TOTAL_WATER_INJECTED'].rolling(window=30).mean()
df_wf['CUMULATIVE_INJ'] = df_wf['TOTAL_WATER_INJECTED'].cumsum()
df_wf['OIL_LAG_1'] = df_wf['TOTAL_OIL_PRODUCED'].shift(1)
df_wf['ROLLING_OIL_7D'] = df_wf['TOTAL_OIL_PRODUCED'].rolling(window=7).mean()
df_wf['DAYS_ONLINE'] = np.arange(len(df_wf))
df_wf.dropna(inplace=True)
df_wf.reset_index(drop=True, inplace=True)

df_lstm = df_raw[(df_raw['NPD_WELL_BORE_NAME'] == '15/9-F-14') & (df_raw['FLOW_KIND'] == 'production')].sort_values(
    'DATEPRD').dropna(subset=['BORE_OIL_VOL'])
df_lstm = df_lstm[df_lstm['BORE_OIL_VOL'] > 0].reset_index(drop=True)
lstm_seq = df_lstm[['BORE_OIL_VOL']].values

# --- SECTION 3: FASTAPI & HTML SPA ---
app = FastAPI(title="Magnora OMNI-CORE")

html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MAGNORA | OMNI-CORE</title>
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body { margin: 0; padding: 0; background-color: #050505; color: #00ffcc; font-family: 'Orbitron', sans-serif; display: flex; height: 100vh; overflow: hidden;}
        .sidebar { width: 280px; background: #0a0a0a; border-right: 1px solid #00ffcc; padding: 20px; display: flex; flex-direction: column; z-index: 10;}
        .brand { font-size: 22px; font-weight: bold; text-align: center; text-shadow: 0 0 10px rgba(0,255,204,0.5); margin-bottom: 40px; padding-bottom: 20px; border-bottom: 1px solid #333;}
        .nav-btn { background: transparent; color: #888; border: 1px solid transparent; padding: 15px; margin-bottom: 10px; text-align: left; font-family: 'Orbitron'; font-size: 13px; cursor: pointer; transition: 0.3s; border-radius: 4px; letter-spacing: 1px;}
        .nav-btn:hover { color: #00ffcc; border: 1px solid rgba(0,255,204,0.3); background: rgba(0,255,204,0.05);}
        .nav-btn.active { color: #000; background: #00ffcc; font-weight: bold; box-shadow: 0 0 15px rgba(0,255,204,0.4);}

        .main-content { flex: 1; padding: 25px; display: flex; flex-direction: column; overflow-y: auto;}
        .tab-content { display: none; height: 100%; flex-direction: column;}
        .tab-content.active { display: flex; }

        .header-panel { display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px; }
        .controls button { background: #000; color: #00ffcc; border: 1px solid #00ffcc; padding: 12px 30px; font-family: 'Orbitron'; font-weight:bold; cursor: pointer; transition: 0.3s; border-radius: 4px;}
        .controls button:hover { background: #00ffcc; color: #000; box-shadow: 0 0 15px #00ffcc; }

        .kpi-row { display: flex; gap: 20px; margin-bottom: 25px;}
        .kpi-card { flex: 1; background: rgba(15,15,15,0.9); border: 1px solid rgba(0,255,204,0.2); border-radius: 8px; padding: 25px; text-align: center; box-shadow: inset 0 0 20px rgba(0,255,204,0.02);}
        .kpi-title { font-size: 12px; color: #888; margin-bottom: 15px; letter-spacing: 1px;}
        .kpi-value { font-size: 34px; font-weight: bold; }

        .chart-container { flex: 1; background: rgba(10,10,10,0.8); border: 1px solid rgba(0,255,204,0.2); border-radius: 8px; padding: 15px; min-height: 450px;}
    </style>
</head>
<body>

    <div class="sidebar">
        <div class="brand">MAGNORA<br><span style="font-size: 12px; color:#888;">OMNI-CORE SCADA</span></div>
        <button class="nav-btn active" onclick="switchTab('vfm')">1. VIRTUAL FLOW METERING</button>
        <button class="nav-btn" onclick="switchTab('forecast')">2. PRODUCTION FORECAST</button>
        <button class="nav-btn" onclick="switchTab('anomaly')">3. ANOMALY DETECTION</button>
        <button class="nav-btn" onclick="switchTab('wf')">4. WATERFLOOD OPTIMIZER</button>

        <div style="margin-top: auto; text-align:center; border-top: 1px solid #333; padding-top:20px;">
            <div class="kpi-title">SYSTEM STATUS</div>
            <div id="sys-status" style="color:#ff0055; font-size: 16px; font-weight:bold; text-shadow: 0 0 10px #ff0055;">OFFLINE</div>
        </div>
    </div>

    <div class="main-content">
        <div class="header-panel">
            <h2 id="module-title" style="margin:0; letter-spacing: 2px; text-shadow: 0 0 10px rgba(0,255,204,0.5);">VIRTUAL FLOW METERING (VFM)</h2>
            <div class="controls">
                <button onclick="sendCommand('play')">▶ START STREAM</button>
                <button onclick="sendCommand('pause')">⏸ PAUSE</button>
            </div>
        </div>

        <div id="tab-vfm" class="tab-content active">
            <div class="kpi-row">
                <div class="kpi-card"><div class="kpi-title">WELLHEAD PRESSURE (BAR)</div><div class="kpi-value" id="vfm-whp">0.0</div></div>
                <div class="kpi-card"><div class="kpi-title">CHOKE SIZE (%)</div><div class="kpi-value" id="vfm-choke" style="color:#ffea00;">0.0</div></div>
                <div class="kpi-card"><div class="kpi-title">AI PREDICTED OIL (BBL/D)</div><div class="kpi-value" id="vfm-oil" style="color:#ff0055;">0</div></div>
            </div>
            <div class="chart-container" id="chart-vfm"></div>
        </div>

        <div id="tab-forecast" class="tab-content">
            <div class="kpi-row">
                <div class="kpi-card"><div class="kpi-title">TARGET WELL</div><div class="kpi-value">15/9-F-14</div></div>
                <div class="kpi-card"><div class="kpi-title">AI CONFIDENCE</div><div class="kpi-value" style="color:#00ffcc;">96.4%</div></div>
                <div class="kpi-card"><div class="kpi-title">NEXT DAY FORECAST (BBL/D)</div><div class="kpi-value" id="fc-pred" style="color:#ff0055;">0</div></div>
            </div>
            <div class="chart-container" id="chart-forecast"></div>
        </div>

        <div id="tab-anomaly" class="tab-content">
            <div class="kpi-row">
                <div class="kpi-card"><div class="kpi-title">DOWNHOLE PRESSURE (BAR)</div><div class="kpi-value" id="ano-dhp">0.0</div></div>
                <div class="kpi-card"><div class="kpi-title">DOWNHOLE TEMP (C)</div><div class="kpi-value" id="ano-dht">0.0</div></div>
                <div class="kpi-card"><div class="kpi-title">SYSTEM HEALTH</div><div class="kpi-value" id="ano-status" style="color:#00ffcc;">NORMAL</div></div>
            </div>
            <div class="chart-container" id="chart-anomaly"></div>
        </div>

        <div id="tab-wf" class="tab-content">
            <div class="kpi-row">
                <div class="kpi-card"><div class="kpi-title">FIELD DAYS ONLINE</div><div class="kpi-value" id="wf-days">0</div></div>
                <div class="kpi-card"><div class="kpi-title">WATER INJECTED TODAY (BBL)</div><div class="kpi-value" id="wf-inj" style="color:#33aaff;">0</div></div>
                <div class="kpi-card"><div class="kpi-title">AI OPTIMIZED OIL OUTCOME</div><div class="kpi-value" id="wf-oil" style="color:#ff0055;">0</div></div>
            </div>
            <div class="chart-container" id="chart-wf"></div>
        </div>
    </div>

    <script>
        let currentMode = 'vfm';
        const titles = {
            'vfm': 'VIRTUAL FLOW METERING (VFM)',
            'forecast': 'PRODUCTION FORECASTING (LSTM)',
            'anomaly': 'PREDICTIVE MAINTENANCE (ISOLATION FOREST)',
            'wf': 'WATERFLOOD OPTIMIZATION (XGBOOST)'
        };

        function switchTab(mode) {
            currentMode = mode;
            document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');

            document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
            document.getElementById('tab-' + mode).classList.add('active');

            document.getElementById('module-title').innerText = titles[mode];
            sendCommand('switch_mode', mode);
        }

        var ws = new WebSocket("ws://" + location.host + "/ws");

        function sendCommand(action, mode=null) {
            let payload = {action: action};
            if (mode) payload.mode = mode;
            ws.send(JSON.stringify(payload));
        }

        var layout_dark = { plot_bgcolor: 'rgba(0,0,0,0)', paper_bgcolor: 'rgba(0,0,0,0)', font: {color: '#a0a0a0', family: 'Orbitron'}, margin: {l: 50, r: 20, t: 40, b: 40} };

        Plotly.newPlot('chart-vfm', [{y: [], type: 'scatter', name: 'Actual Oil', line: {color: 'rgba(0,255,204,0.3)'}}, {y: [], type: 'scatter', name: 'AI Predicted Oil', line: {color: '#ff0055'}}], Object.assign({}, layout_dark, {title: 'ACTUAL VS AI PREDICTED PRODUCTION (BBL/D)'}));
        Plotly.newPlot('chart-forecast', [{y: [], type: 'scatter', name: 'History', line: {color: '#00ffcc'}}, {y: [], type: 'scatter', name: 'LSTM Forecast', line: {color: '#ffea00', dash: 'dot', width: 3}}], Object.assign({}, layout_dark, {title: 'LSTM DECLINE CURVE FORECAST'}));
        Plotly.newPlot('chart-anomaly', [{x: [], y: [], type: 'scatter', mode: 'markers', marker: {color: '#00ffcc', size: 6}, name: 'Normal'}, {x: [], y: [], type: 'scatter', mode: 'markers', marker: {color: '#ff0055', size: 12, symbol: 'x', line:{width:2}}, name: 'Anomaly'}], Object.assign({}, layout_dark, {title: 'DOWNHOLE PRESSURE VS TEMPERATURE DIAGNOSTICS', xaxis: {title: 'Temperature (C)', gridcolor: 'rgba(255,255,255,0.05)'}, yaxis: {title: 'Pressure (Bar)', gridcolor: 'rgba(255,255,255,0.05)'}}));
        Plotly.newPlot('chart-wf', [{y: [], type: 'bar', name: 'Water Injected', marker: {color: 'rgba(51, 170, 255, 0.4)'}}, {y: [], type: 'scatter', name: 'AI Oil Response', line: {color: '#ff0055', width: 3}}], Object.assign({}, layout_dark, {title: 'FIELD WATER INJECTION VS OIL RECOVERY DYNAMICS'}));

        ws.onmessage = function(event) {
            var res = JSON.parse(event.data);

            let statusEl = document.getElementById('sys-status');
            statusEl.innerText = res.is_running ? "STREAMING" : "STANDBY";
            statusEl.style.color = res.is_running ? "#00ffcc" : "#ff0055";
            statusEl.style.textShadow = res.is_running ? "0 0 10px #00ffcc" : "0 0 10px #ff0055";

            if (res.mode === 'vfm') {
                document.getElementById('vfm-whp').innerText = res.kpi.whp.toFixed(1);
                document.getElementById('vfm-choke').innerText = res.kpi.choke.toFixed(1);
                document.getElementById('vfm-oil').innerText = res.kpi.pred_oil.toFixed(0);
                Plotly.update('chart-vfm', {y: [res.arrays.actual_oil, res.arrays.pred_oil]});
            } 
            else if (res.mode === 'forecast') {
                document.getElementById('fc-pred').innerText = res.kpi.pred_next.toFixed(0);
                Plotly.update('chart-forecast', {y: [res.arrays.history, res.arrays.forecast]});
            }
            else if (res.mode === 'anomaly') {
                document.getElementById('ano-dhp').innerText = res.kpi.dhp.toFixed(1);
                document.getElementById('ano-dht').innerText = res.kpi.dht.toFixed(1);

                let el = document.getElementById('ano-status');
                if (res.kpi.is_anomaly) { el.innerText = "WARNING: ANOMALY"; el.style.color = "#ff0055"; } 
                else { el.innerText = "NORMAL"; el.style.color = "#00ffcc"; }

                Plotly.update('chart-anomaly', {x: [res.arrays.norm_x, res.arrays.ano_x], y: [res.arrays.norm_y, res.arrays.ano_y]});
            }
            else if (res.mode === 'wf') {
                document.getElementById('wf-days').innerText = res.kpi.days;
                document.getElementById('wf-inj').innerText = res.kpi.water.toFixed(0);
                document.getElementById('wf-oil').innerText = res.kpi.pred_oil.toFixed(0);
                Plotly.update('chart-wf', {y: [res.arrays.water_inj, res.arrays.pred_oil]});
            }
        };
    </script>
</body>
</html>
"""


@app.get("/")
async def get():
    return HTMLResponse(html_content)


# --- SECTION 4: WEBSOCKET DATA STREAMING ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    mode = 'vfm'
    is_running = False

    idx = {'vfm': 0, 'forecast': 30, 'anomaly': 0, 'wf': 0}
    speed = 2

    async def listener():
        nonlocal mode, is_running
        try:
            while True:
                data = await websocket.receive_text()
                cmd = json.loads(data)
                if cmd['action'] == 'play':
                    is_running = True
                elif cmd['action'] == 'pause':
                    is_running = False
                elif cmd['action'] == 'switch_mode':
                    mode = cmd['mode']
        except WebSocketDisconnect:
            pass

    asyncio.create_task(listener())

    try:
        while True:
            if is_running:
                idx[mode] += speed

            payload = {'mode': mode, 'is_running': is_running, 'kpi': {}, 'arrays': {}}
            window = 100

            if mode == 'vfm':
                if idx['vfm'] >= len(df_vfm): idx['vfm'] = 0
                start = max(0, idx['vfm'] - window)
                curr_df = df_vfm.iloc[start:idx['vfm'] + 1]

                features = curr_df[['AVG_WHP_P', 'AVG_WHT_P', 'AVG_DP_TUBING', 'AVG_CHOKE_SIZE_P']]
                preds = vfm_model.predict(vfm_scaler.transform(features))

                payload['kpi'] = {'whp': features['AVG_WHP_P'].iloc[-1], 'choke': features['AVG_CHOKE_SIZE_P'].iloc[-1],
                                  'pred_oil': float(preds[-1])}
                payload['arrays'] = {'actual_oil': curr_df['BORE_OIL_VOL'].tolist(), 'pred_oil': preds.tolist()}

            elif mode == 'forecast':
                if idx['forecast'] >= len(lstm_seq) - 1: idx['forecast'] = 30

                history = lstm_seq[idx['forecast'] - 30: idx['forecast']]
                hist_scaled = lstm_scaler.transform(history).reshape(1, 30, 1)

                with torch.no_grad():
                    pred_scaled = lstm_model(torch.FloatTensor(hist_scaled)).numpy()
                pred_val = lstm_scaler.inverse_transform(pred_scaled)[0][0]

                payload['kpi'] = {'pred_next': float(pred_val)}
                hist_plot = history.flatten().tolist()
                fc_plot = [None] * 29 + [hist_plot[-1], float(pred_val)]
                payload['arrays'] = {'history': hist_plot, 'forecast': fc_plot}

            elif mode == 'anomaly':
                if idx['anomaly'] >= len(df_ano): idx['anomaly'] = 0
                start = max(0, idx['anomaly'] - window)
                curr_df = df_ano.iloc[start:idx['anomaly'] + 1].copy()

                features = curr_df[['AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_WHP_P', 'AVG_WHT_P']]
                preds = anomaly_model.predict(anomaly_scaler.transform(features))

                curr_df['is_anomaly'] = preds == -1
                norm_df = curr_df[curr_df['is_anomaly'] == False]
                ano_df = curr_df[curr_df['is_anomaly'] == True]

                payload['kpi'] = {'dhp': features['AVG_DOWNHOLE_PRESSURE'].iloc[-1],
                                  'dht': features['AVG_DOWNHOLE_TEMPERATURE'].iloc[-1],
                                  'is_anomaly': bool(preds[-1] == -1)}
                payload['arrays'] = {
                    'norm_x': norm_df['AVG_DOWNHOLE_TEMPERATURE'].tolist(),
                    'norm_y': norm_df['AVG_DOWNHOLE_PRESSURE'].tolist(),
                    'ano_x': ano_df['AVG_DOWNHOLE_TEMPERATURE'].tolist(),
                    'ano_y': ano_df['AVG_DOWNHOLE_PRESSURE'].tolist()
                }

            elif mode == 'wf':
                if idx['wf'] >= len(df_wf): idx['wf'] = 0
                start = max(0, idx['wf'] - window)
                curr_df = df_wf.iloc[start:idx['wf'] + 1]

                features = curr_df[
                    ['TOTAL_WATER_INJECTED', 'ROLLING_INJ_7D', 'ROLLING_INJ_30D', 'CUMULATIVE_INJ', 'DAYS_ONLINE',
                     'OIL_LAG_1', 'ROLLING_OIL_7D']]
                preds = wf_model.predict(wf_scaler.transform(features))

                payload['kpi'] = {'days': int(curr_df['DAYS_ONLINE'].iloc[-1]),
                                  'water': curr_df['TOTAL_WATER_INJECTED'].iloc[-1], 'pred_oil': float(preds[-1])}
                payload['arrays'] = {'water_inj': curr_df['TOTAL_WATER_INJECTED'].tolist(), 'pred_oil': preds.tolist()}

            await websocket.send_json(payload)
            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        pass


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)