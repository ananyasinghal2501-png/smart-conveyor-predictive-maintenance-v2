# Smart Conveyor Predictive Maintenance System

An IoT and machine-learning prototype for monitoring conveyor belt health, detecting known faults, and flagging unusual sensor behaviour.

## Technology Stack

- **Hardware:** ESP32 and conveyor sensors
- **Backend:** Python, FastAPI
- **Machine Learning:** Random Forest classifier and Isolation Forest anomaly detector
- **Frontend:** HTML, CSS, JavaScript, Plotly and 3D digital-twin visualisation
- **Database:** SQLite
- **Communication:** REST API and WebSocket

## System Features

- Classification of healthy, joint defect, overheat and rupture conditions
- Unsupervised anomaly detection for unfamiliar sensor patterns
- Temporal analysis of consecutive sensor readings
- Live sensor graphs, health score, fault status and alerts
- Independent ESP32 flame-safety logic

## Setup (Windows)

1. Install Python and open this project in VS Code.
2. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. If model files need to be regenerated, run:

   ```powershell
   python ml/train_model.py
   python ml/train_anomaly_model.py
   ```

5. Start the backend:

   ```powershell
   python -m uvicorn main:app --host 0.0.0.0 --port 8000
   ```

6. Test the API at `http://127.0.0.1:8000/docs`.

7. Open `index.html` using the VS Code Live Server extension to view the dashboard.

## Important Notes

- The Random Forest recognises four known classes.
- The Isolation Forest flags patterns that differ from its learned healthy baseline; an anomaly is not automatically a confirmed fault.
- Temporal analysis requires consecutive sensor readings.
- Real-time hardware monitoring requires the ESP32 to send sensor data to the backend.
- Never commit Wi-Fi passwords, `.env` secrets or private credentials.
- Anomaly detection is experimental and requires validation on fresh sensor readings.
