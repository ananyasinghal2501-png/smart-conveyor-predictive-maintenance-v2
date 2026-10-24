import pandas as pd
import joblib
from pathlib import Path


# ==========================================
# EXPECTED FEATURES
# ==========================================

FEATURES = [
    "object_temperature",
    "ambient_temperature",
    "vibration",
    "alignment",
    "load",
    "speed",
    "flame"
]


# ==========================================
# MODEL PATH
# ==========================================

MODEL_PATH = Path(__file__).parent / "conveyor_model.pkl"


print("\nCONVEYOR FAULT PREDICTION SYSTEM")
print("---------------------------------")


# ==========================================
# TAKE SENSOR VALUES
# ==========================================

object_temperature = float(
    input("Enter object temperature (°C): ")
)

ambient_temperature = float(
    input("Enter ambient temperature (°C): ")
)

vibration = float(
    input("Enter vibration magnitude (Mag): ")
)

alignment = float(
    input("Enter alignment deviation (Dev): ")
)

load = float(
    input("Enter load (g): ")
)

speed = float(
    input("Enter speed (0-100): ")
)

flame = int(
    input("Enter flame status (1 = Safe, 0 = Flame Detected): ")
)


# ==========================================
# DIRECT FIRE SAFETY CHECK
# ==========================================

if flame == 0:

    print("\n-----------------------------")
    print("CONVEYOR STATUS")
    print("-----------------------------")

    print("Predicted Condition: Fire")
    print("Confidence: 100.00%")
    print("Severity: CRITICAL")
    print("Action: Conveyor should be stopped")

    raise SystemExit


# ==========================================
# CREATE INPUT DATA
# ==========================================

sensor_data = pd.DataFrame([{
    "object_temperature": object_temperature,
    "ambient_temperature": ambient_temperature,
    "vibration": vibration,
    "alignment": alignment,
    "load": load,
    "speed": speed,
    "flame": flame
}])


# ==========================================
# TRY TO LOAD MODEL
# ==========================================

try:

    model = joblib.load(MODEL_PATH)

    # Check whether this is the NEW model
    model_features = list(
        getattr(model, "feature_names_in_", [])
    )


    if model_features != FEATURES:

        print("\n-----------------------------")
        print("DATA COLLECTION MODE")
        print("-----------------------------")

        print(
            "The saved ML model is still the old model."
        )

        print(
            "New fault/rupture data is required before retraining."
        )

        print("\nCurrent sensor reading:")

        print(
            sensor_data.to_string(index=False)
        )


    else:

        # ==================================
        # ML PREDICTION
        # ==================================

        prediction = model.predict(
            sensor_data
        )[0]

        probabilities = model.predict_proba(
            sensor_data
        )[0]

        confidence = (
            probabilities.max() * 100
        )


        print("\n-----------------------------")
        print("CONVEYOR AI PREDICTION")
        print("-----------------------------")

        print(
            f"Predicted Condition: {prediction}"
        )

        print(
            f"Confidence: {confidence:.2f}%"
        )


except Exception as error:

    print("\n-----------------------------")
    print("DATA COLLECTION MODE")
    print("-----------------------------")

    print(
        "ML model is not ready yet."
    )

    print(
        "Collect fault/rupture readings before training."
    )

    print("\nCurrent sensor reading:")

    print(
        sensor_data.to_string(index=False)
    )