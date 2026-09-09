import pandas as pd
import joblib


# Load the trained model
model = joblib.load("conveyor_model.pkl")

print("Conveyor Fault Prediction System")
print("--------------------------------")


# Take sensor readings from the user
temperature = float(input("Enter temperature (°C): "))
vibration = float(input("Enter vibration: "))
gyro = float(input("Enter gyro reading: "))
flame = int(input("Enter flame status (0 = No, 1 = Yes): "))


# Create input data
sensor_data = pd.DataFrame([{
    "temperature": temperature,
    "vibration": vibration,
    "gyro": gyro,
    "flame": flame
}])


# Make prediction
prediction = model.predict(sensor_data)[0]


# Get prediction probabilities
probabilities = model.predict_proba(sensor_data)[0]

classes = model.classes_

confidence = probabilities.max() * 100


# Display result
print("\n-----------------------------")
print("CONVEYOR AI PREDICTION")
print("-----------------------------")

print(f"Predicted Condition: {prediction}")
print(f"Confidence: {confidence:.2f}%")