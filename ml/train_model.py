import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# FILE PATHS
# ==========================================

BASE_DIR = Path(__file__).parent

DATASET_PATH = BASE_DIR / "dataset.csv"
MODEL_PATH = BASE_DIR / "conveyor_model.pkl"


# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully")
print("Total rows:", len(df))

print("\nClass distribution:")
print(df["fault"].value_counts())


# ==========================================
# FEATURES
# ==========================================

FEATURES = [
    "object_temperature",
    "ambient_temperature",
    "vibration",
    "alignment",
    "load"
]

TARGET = "fault"

X = df[FEATURES]
y = df[TARGET]


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# RANDOM FOREST
# ==========================================

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)


# ==========================================
# MODEL TEST
# ==========================================

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n================================")
print("MODEL RESULTS")
print("================================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# ==========================================
# FEATURE IMPORTANCE
# ==========================================

print("\nFeature Importance:")

importance_table = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

print(
    importance_table.to_string(index=False)
)


# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    model,
    MODEL_PATH
)

print("\n================================")
print("MODEL SAVED SUCCESSFULLY")
print("================================")

print("Saved at:")
print(MODEL_PATH)

print("\nClasses learned:")
print(model.classes_)
