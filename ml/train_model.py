import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


# -----------------------------------
# 1. Load the dataset
# -----------------------------------

data = pd.read_csv("dataset.csv")

print("Dataset loaded successfully!")
print(data.head())


# -----------------------------------
# 2. Separate inputs and output
# -----------------------------------

X = data[
    [
        "temperature",
        "vibration",
        "gyro",
        "flame"
    ]
]

y = data["fault"]


# -----------------------------------
# 3. Split data into training/testing
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------------
# 4. Create the Random Forest model
# -----------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# -----------------------------------
# 5. Train the model
# -----------------------------------

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed!")


# -----------------------------------
# 6. Test the model
# -----------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# -----------------------------------
# 7. Save the trained model
# -----------------------------------

joblib.dump(model, "conveyor_model.pkl")

print("\nModel saved successfully!")
print("File: conveyor_model.pkl")