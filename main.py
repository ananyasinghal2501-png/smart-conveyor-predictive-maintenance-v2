from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
from collections import deque
from statistics import median

from database import SessionLocal, engine, Base
from models import SensorReading

import pandas as pd
import joblib


# ==========================================
# CREATE DATABASE TABLES
# ==========================================

Base.metadata.create_all(bind=engine)


# ==========================================
# FASTAPI APPLICATION
# ==========================================

app = FastAPI()
# ==========================================
# WEBSOCKET CONNECTION MANAGER
# ==========================================

class ConnectionManager:

    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, data):
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception:
                disconnected.append(connection)

        for connection in disconnected:
            self.disconnect(connection)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    await manager.connect(websocket)

    try:

        # Send current state immediately when dashboard connects
        await websocket.send_json(current_state)

        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:

        manager.disconnect(websocket)


# Allow frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# LOAD ML MODEL
# ==========================================
# ==========================================
# LOAD TRAINED ML MODEL
# ==========================================

MODEL_PATH = (
    Path(__file__).parent
    / "ml"
    / "conveyor_model.pkl"
)

if MODEL_PATH.exists():

    model = joblib.load(MODEL_PATH)

    print("================================")
    print("RANDOM FOREST MODEL LOADED")
    print("================================")
    print("Model path:", MODEL_PATH)
    print("Classes:", model.classes_)

else:

    model = None

    print("WARNING: conveyor_model.pkl not found")

ANOMALY_MODEL_PATH = Path(__file__).parent / "ml" / "anomaly_model.pkl"
anomaly_model = joblib.load(ANOMALY_MODEL_PATH) if ANOMALY_MODEL_PATH.exists() else None
if anomaly_model is None:
    print("Anomaly model not found yet; train ml/train_anomaly_model.py")


# ==========================================
# SENSOR DATA STRUCTURE
# ==========================================

class SensorData(BaseModel):

    # Thermal sensor
    object_temperature: float
    ambient_temperature: float

    # Gyro / accelerometer module
    vibration: float
    alignment: float

    # Load cell
    load: float

    # Motor speed
    speed: float

    # Flame sensor
    # 1 = SAFE
    # 0 = FLAME DETECTED
    flame: int


# ==========================================
# CURRENT CONVEYOR STATE
# ==========================================

current_state = {

    "object_temperature": 0,
    "ambient_temperature": 0,

    "vibration": 0,
    "alignment": 0,
    "load": 0,
    "speed": 0,
    "flame": 1,

    "prediction": "Unknown",
    "confidence": 0,

    "severity": "Unknown",
    "health_score": 0,

    "timestamp": None,

    "why_status": [
        "Waiting for sensor data"
    ]
}


# ==========================================
# TEMPORARY SENSOR HISTORY
# ==========================================

sensor_history = []

# TEMPORAL ANALYSIS ADD-ON

from collections import deque
from statistics import median

TEMPORAL_FEATURES = ["object_temperature", "ambient_temperature", "vibration", "alignment", "load"]
TEMPORAL_FLOORS = {"object_temperature": 0.5, "ambient_temperature": 0.5, "vibration": 0.05, "alignment": 0.05, "load": 2.0}
temporal_history = deque(maxlen=20)

def update_temporal_analysis(data):
    current = {k: float(getattr(data, k)) for k in TEMPORAL_FEATURES}
    previous = list(temporal_history)
    if len(previous) < 10:
        temporal_history.append(current)
        return {"ready": False, "samples": len(temporal_history), "status": "Warming up",
                "shift_detected": False, "scores": {}, "rolling_features": {}}
    scores, rolling = {}, {}
    for key in TEMPORAL_FEATURES:
        vals = [row[key] for row in previous]
        centre = median(vals)
        mad = median([abs(v - centre) for v in vals])
        scale = max(1.4826 * mad, TEMPORAL_FLOORS[key])
        scores[key] = round(abs(current[key] - centre) / scale, 3)
        rolling[key + "_median"] = round(centre, 4)
        rolling[key + "_mean"] = round(sum(vals) / len(vals), 4)
        rolling[key + "_range"] = round(max(vals) - min(vals), 4)
    suspicious = [k for k, v in scores.items() if v >= 4.0]
    shift = len(suspicious) >= 2 or scores["vibration"] >= 6.0 or scores["alignment"] >= 6.0
    temporal_history.append(current)
    return {"ready": True, "samples": len(previous), "status": "Shift detected" if shift else "Monitoring",
            "shift_detected": shift, "scores": scores, "rolling_features": rolling,
            "suspicious_features": suspicious}



# ==========================================
# TEMPORARY ALERT HISTORY
# ==========================================

alerts = []


# ==========================================
# WHY THIS STATUS?
# ==========================================
def generate_status_explanation(
    object_temperature,
    ambient_temperature,
    vibration,
    alignment,
    load,
    speed,
    flame,
    prediction,
    severity
):

    reasons = []


    # ======================================
    # TEMPERATURE
    # ======================================

    if object_temperature >= 80:

        reasons.append(
            "Object temperature is critically high"
        )

    elif object_temperature >= 70:

        reasons.append(
            "Object temperature is above the normal range"
        )

    elif object_temperature >= 60:

        reasons.append(
            "Object temperature is slightly elevated"
        )


    # ======================================
    # VIBRATION
    # ======================================

    if vibration >= 8:

        reasons.append(
            "Vibration is critically high"
        )

    elif vibration >= 5:

        reasons.append(
            "Vibration is above the normal range"
        )

    elif vibration >= 3:

        reasons.append(
            "Vibration is elevated"
        )


    # ======================================
    # ALIGNMENT
    # ======================================

    alignment_deviation = abs(alignment)

    if alignment_deviation >= 1.0:

        reasons.append(
            "Severe belt alignment deviation detected"
        )

    elif alignment_deviation >= 0.5:

        reasons.append(
            "Belt alignment is outside the normal range"
        )

    elif alignment_deviation >= 0.3:

        reasons.append(
            "Belt alignment is drifting"
        )


    # ======================================
    # MOTOR / SPEED STATE
    # ======================================

    if speed == 0:

        if flame == 0:

            reasons.append(
                "Conveyor stopped automatically after flame detection"
            )

        else:

            reasons.append(
                "Conveyor belt is stopped"
            )


    # ======================================
    # FLAME SENSOR
    # ======================================

    if flame == 0:

        reasons.append(
            "Flame detected by the flame sensor"
        )


    # ======================================
    # ML PREDICTION
    # ======================================

    if prediction == "joint_defect":

        reasons.append(
            "ML detected a joint defect pattern"
        )

    elif prediction == "overheat":

        reasons.append(
            "ML detected an overheating pattern"
        )

    elif prediction == "rupture":

        reasons.append(
            "ML detected a rupture pattern"
        )

    elif prediction == "healthy":

        reasons.append(
            "ML classifies the conveyor condition as healthy"
        )


    # ======================================
    # SEVERITY
    # ======================================

    if severity == "Critical":

        reasons.append(
            "Critical safety condition detected"
        )

    elif severity == "Warning":

        reasons.append(
            "Warning condition detected"
        )


    # ======================================
    # FALLBACK
    # ======================================

    if not reasons:

        reasons.append(
            "All monitored parameters are within the normal range"
        )


    return reasons



# ==========================================
# HOME
# ==========================================

@app.get("/")
def home():

    return {
        "message":
        "Conveyor Monitoring Backend is running!"
    }


# ==========================================
# RECEIVE SENSOR DATA
# ==========================================
@app.post("/sensor-data")
async def receive_sensor_data(data: SensorData):

    global current_state

    # ======================================
    # TIMESTAMP
    # ======================================

    timestamp = datetime.now().isoformat()

    anomaly_input = pd.DataFrame([{
        "object_temperature": data.object_temperature,
        "ambient_temperature": data.ambient_temperature,
        "vibration": data.vibration,
        "alignment": data.alignment,
        "load": data.load,
    }])
    temporal_analysis = update_temporal_analysis(data)
    anomaly_detected = False
    anomaly_score = None
    anomaly_status = "Model unavailable"
    if anomaly_model is not None:
        try:
            anomaly_detected = int(anomaly_model.predict(anomaly_input)[0]) == -1
            anomaly_score = float(anomaly_model.decision_function(anomaly_input)[0])
        except Exception as error:
            print("Anomaly detection error:", error)
            anomaly_status = "Model error"
    if temporal_analysis["shift_detected"]:
        anomaly_detected = True
    if anomaly_detected:
        anomaly_status = "Investigate"
    elif anomaly_model is not None:
        anomaly_status = "Warming up" if not temporal_analysis["ready"] else "No anomaly detected"
    elif temporal_analysis["ready"]:
        anomaly_status = "Temporal monitoring only"
    else:
        anomaly_status = "Warming up"



    # ======================================
    # EMERGENCY FLAME OVERRIDE
    # ======================================

    # Flame remains independent of ML.
    # 0 = flame detected
    # 1 = safe

    if data.flame == 0:

        prediction = "Fire"
        confidence = 100.0
        severity = "Critical"

    else:

        # ==================================
        # RANDOM FOREST PREDICTION
        # ==================================

        if model is not None:

            input_data = pd.DataFrame([{
                "object_temperature": data.object_temperature,
                "ambient_temperature": data.ambient_temperature,
                "vibration": data.vibration,
                "alignment": data.alignment,
                "load": data.load
            }])

            prediction = model.predict(input_data)[0]

            probabilities = model.predict_proba(input_data)[0]

            confidence = float(
                probabilities.max() * 100
            )


            # ==================================
            # ML CLASS → SEVERITY
            # ==================================

            if prediction == "healthy":

                severity = "Normal"

            elif prediction == "joint_defect":

                severity = "Warning"

            elif prediction == "overheat":

                if data.object_temperature >= 80:
                    severity = "Critical"
                else:
                    severity = "Warning"

            elif prediction == "rupture":

                severity = "Critical"

            else:

                severity = "Warning"

        else:

            prediction = "Model_Not_Loaded"
            confidence = 0.0
            severity = "Warning"


    # ======================================
    if data.flame != 0 and anomaly_detected and severity == "Normal":
        severity = "Warning"

    # DYNAMIC HEALTH SCORE
    # ======================================

    if prediction == "Fire":

        health_score = 0

    elif model is not None:

        probability_map = {
            str(label): float(probability)
            for label, probability in zip(
                model.classes_,
                probabilities
            )
        }

        class_health = {
            "healthy": 100,
            "joint_defect": 65,
            "overheat": 35,
            "rupture": 10
        }

        health_score = round(
            sum(
                probability_map.get(label, 0) * score
                for label, score in class_health.items()
            )
        )

        health_score = max(
            0,
            min(100, health_score)
        )

    else:

        health_score = 50
    # ======================================
    # WHY THIS STATUS?
    # ======================================

    why_status = generate_status_explanation(
        data.object_temperature,
        data.ambient_temperature,
        data.vibration,
        data.alignment,
        data.load,
        data.speed,
        data.flame,
        prediction,
        severity
    )


    # ======================================
    if anomaly_detected:
        why_status.append("ANOMALY WARNING: unusual sensor behaviour; investigate. This is not a confirmed fault.")

    # UPDATE CURRENT STATE
    # ======================================

    current_state = {
        "object_temperature": data.object_temperature,
        "ambient_temperature": data.ambient_temperature,
        "vibration": data.vibration,
        "alignment": data.alignment,
        "load": data.load,
        "speed": data.speed,
        "flame": data.flame,
        "prediction": str(prediction),
        "confidence": round(confidence, 2),
        "severity": severity,
        "health_score": health_score,
        "timestamp": timestamp,
        "why_status": why_status,
        "anomaly_detected": anomaly_detected,
        "anomaly_status": anomaly_status,
        "anomaly_score": round(anomaly_score, 5) if anomaly_score is not None else None,
        "temporal_analysis": temporal_analysis
    }
        # Send latest state instantly to connected dashboards
    await manager.broadcast(current_state)


    # ======================================
    # SAVE FOR TREND GRAPHS
    # ======================================

    sensor_history.append(current_state.copy())

    if len(sensor_history) > 1000:
        del sensor_history[:-1000]


    # ======================================
    # SAVE TO DATABASE
    # ======================================

    db = SessionLocal()

    try:

        db_reading = SensorReading(
            timestamp=datetime.now(),
            object_temperature=data.object_temperature,
            ambient_temperature=data.ambient_temperature,
            vibration=data.vibration,
            alignment=data.alignment,
            load=data.load,
            speed=data.speed,
            flame=data.flame,
            prediction=str(prediction),
            confidence=round(confidence, 2),
            severity=severity,
            health_score=health_score
        )

        db.add(db_reading)
        db.commit()

    except Exception as e:

        db.rollback()

        print("Database error:", e)

    finally:

        db.close()


    # ======================================
    # CREATE ALERT
    # ======================================

    if severity != "Normal":

        alert = {
            "timestamp": timestamp,
            "fault": str(prediction),
            "severity": severity,
            "object_temperature": data.object_temperature,
            "ambient_temperature": data.ambient_temperature,
            "vibration": data.vibration,
            "alignment": data.alignment,
            "load": data.load,
            "speed": data.speed,
            "flame": data.flame
        }

        alerts.append(alert)

        if len(alerts) > 100:
            del alerts[:-100]


    # ======================================
    # RETURN RESULT
    # ======================================

    return current_state


# ==========================================
# GET CURRENT HEALTH
# ==========================================

@app.get("/health")
def get_health():

    return {

        "health_score":
            current_state["health_score"],

        "severity":
            current_state["severity"],


        # Thermal sensor
        "object_temperature":
            current_state["object_temperature"],

        "ambient_temperature":
            current_state["ambient_temperature"],


        # Other hardware readings
        "vibration":
            current_state["vibration"],

        "alignment":
            current_state["alignment"],

        "load":
            current_state["load"],

        "speed":
            current_state["speed"],

        "flame":
            current_state["flame"],


        # Prediction / mode
        "prediction":
            current_state["prediction"],

        "confidence":
            current_state["confidence"],
        "anomaly_detected": current_state.get("anomaly_detected", False),
        "anomaly_status": current_state.get("anomaly_status", "Waiting for readings"),
        "anomaly_score": current_state.get("anomaly_score"),
        "temporal_analysis": current_state.get("temporal_analysis", {}),


        # Explanation
        "why_status":
            current_state["why_status"],


        # Timestamp
        "timestamp":
            current_state["timestamp"]
    }


# ==========================================
# GET ALERTS
# ==========================================

@app.get("/alerts")
def get_alerts():

    return {

        "total_alerts":
            len(alerts),

        "alerts":
            alerts[-10:]
    }


# ==========================================
# GET LATEST PREDICTION
# ==========================================

@app.get("/predictions")
def get_predictions():

    return {

        "prediction":
            current_state["prediction"],

        "confidence":
            current_state["confidence"],

        "severity":
            current_state["severity"],

        "timestamp":
            current_state["timestamp"]
    }


# ==========================================
# GET SENSOR TRENDS
# ==========================================

@app.get("/trends")
def get_trends():

    return {

        "readings":
            sensor_history[-50:]
    }