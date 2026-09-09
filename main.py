from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from database import SessionLocal
from models import SensorReading


app = FastAPI()


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

MODEL_PATH = Path(__file__).parent / "ml" / "conveyor_model.pkl"

model = joblib.load(MODEL_PATH)


# ==========================================
# SENSOR DATA STRUCTURE
# ==========================================

class SensorData(BaseModel):
    temperature: float
    vibration: float
    gyro: float
    flame: int


# ==========================================
# TEMPORARY MEMORY
# ==========================================

# Latest conveyor condition
current_state = {
    "temperature": 0,
    "vibration": 0,
    "gyro": 0,
    "flame": 0,
    "prediction": "Unknown",
    "confidence": 0,
    "severity": "Unknown",
    "health_score": 0,
    "timestamp": None
}


# Store sensor readings for trend graphs
sensor_history = []


# Store alerts
alerts = []
# ==========================================
# WHY THIS STATUS?
# ==========================================

def generate_status_explanation(
    temperature,
    vibration,
    gyro,
    flame,
    prediction,
    severity
):

    reasons = []

    # Temperature
    if temperature >= 80:
        reasons.append("Temperature is critically high")
    elif temperature >= 70:
        reasons.append("Temperature is above the normal range")
    elif temperature >= 60:
        reasons.append("Temperature is slightly elevated")

    # Vibration
    if vibration >= 8:
        reasons.append("Vibration is critically high")
    elif vibration >= 5:
        reasons.append("Vibration is above the normal range")
    elif vibration >= 3:
        reasons.append("Vibration is slightly elevated")

    # Gyro
    if abs(gyro) >= 5:
        reasons.append("Abnormal belt movement detected")
    elif abs(gyro) >= 3:
        reasons.append("Belt movement is increasing")

    # Flame
    if flame == 1:
        reasons.append("Flame detected by the sensor")

    # ML prediction
    if prediction != "Normal":
        reasons.append(
            f"ML detected {prediction.replace('_', ' ')}"
        )

    # Severity
    if severity == "Critical":
        reasons.append("Critical safety threshold crossed")
    elif severity == "Warning":
        reasons.append("Warning threshold crossed")

    # Normal condition
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
        "message": "Conveyor Monitoring Backend is running!"
    }


# ==========================================
# RECEIVE SENSOR DATA + ML PREDICTION
# ==========================================

@app.post("/sensor-data")
def receive_sensor_data(data: SensorData):

    global current_state

    # Current timestamp
    timestamp = datetime.now().isoformat()

    # Convert sensor data into DataFrame
    sensor_data = pd.DataFrame([{
        "temperature": data.temperature,
        "vibration": data.vibration,
        "gyro": data.gyro,
        "flame": data.flame
    }])

    # --------------------------------------
    # ML prediction
    # --------------------------------------

    prediction = model.predict(sensor_data)[0]

    probabilities = model.predict_proba(sensor_data)[0]

    confidence = float(probabilities.max() * 100)


    # --------------------------------------
    # Determine severity
    # --------------------------------------

    if prediction == "Normal":

        severity = "Normal"

    elif prediction in [
        "Overheating",
        "Excessive_Vibration",
        "Belt_Misalignment"
    ]:

        severity = "Warning"

    elif prediction == "Fire":

        severity = "Critical"

    else:

        severity = "Warning"


    # --------------------------------------
    # Health score
    # --------------------------------------

    if severity == "Normal":

        health_score = 90

    elif severity == "Warning":

        health_score = 60

    else:

        health_score = 30
    # Generate WHY THIS STATUS explanation
    # --------------------------------------

    why_status = generate_status_explanation(
        data.temperature,
        data.vibration,
        data.gyro,
        data.flame,
        prediction,
        severity
    )

    # --------------------------------------
    # Update current state
    # --------------------------------------

            # --------------------------------------
    # Update current state
    # --------------------------------------

    current_state = {
        "temperature": data.temperature,
        "vibration": data.vibration,
        "gyro": data.gyro,
        "flame": data.flame,

        "prediction": prediction,

        "confidence": round(confidence, 2),

        "severity": severity,

        "health_score": health_score,

        "timestamp": timestamp,

        "why_status": why_status
    }

    # --------------------------------------
    # Save reading for trends
    # --------------------------------------

    sensor_history.append(current_state.copy())
    # --------------------------------------
    # Save reading to PostgreSQL
    # --------------------------------------

    db = SessionLocal()

    try:
        db_reading = SensorReading(
            timestamp=datetime.now(),
            temperature=data.temperature,
            vibration=data.vibration,
            gyro=data.gyro,
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

    # --------------------------------------
    # Create alert if necessary
    # --------------------------------------

    if severity != "Normal":

        alert = {

            "timestamp": timestamp,

            "fault": prediction,

            "severity": severity,

            "temperature": data.temperature,

            "vibration": data.vibration
        }

        alerts.append(alert)


    # --------------------------------------
    # Return result
    # --------------------------------------

    return current_state


# ==========================================
# GET CURRENT HEALTH
# ==========================================

@app.get("/health")
def get_health():

    return {
        "health_score": current_state["health_score"],
        "severity": current_state["severity"],

        "temperature": current_state["temperature"],
        "vibration": current_state["vibration"],
        "gyro": current_state["gyro"],
        "flame": current_state["flame"],

        "prediction": current_state["prediction"],
        "confidence": current_state["confidence"],

        "why_status": current_state["why_status"],

        "timestamp": current_state["timestamp"]
    }

# ==========================================
# GET ALERTS
# ==========================================

@app.get("/alerts")
def get_alerts():

    return {

        "total_alerts": len(alerts),

        "alerts": alerts[-10:]
    }


# ==========================================
# GET LATEST PREDICTION
# ==========================================

@app.get("/predictions")
def get_predictions():

    return {

        "prediction": current_state["prediction"],

        "confidence": current_state["confidence"],

        "severity": current_state["severity"],

        "timestamp": current_state["timestamp"]
    }


# ==========================================
# GET SENSOR TRENDS
# ==========================================

@app.get("/trends")
def get_trends():

    return {

        "readings": sensor_history[-50:]
    }