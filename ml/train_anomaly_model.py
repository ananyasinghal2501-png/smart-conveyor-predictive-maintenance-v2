from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "dataset.csv")
features = ["object_temperature", "ambient_temperature", "vibration", "alignment", "load"]
healthy = df.loc[df["fault"].astype(str).str.lower().eq("healthy"), features]
healthy = healthy.apply(pd.to_numeric, errors="coerce").dropna()
if len(healthy) < 20:
    raise ValueError(f"Need at least 20 healthy rows; found {len(healthy)}")
model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
model.fit(healthy)
path = BASE / "anomaly_model.pkl"
joblib.dump(model, path)
print(f"Isolation Forest saved: {path}")
print(f"Healthy rows used: {len(healthy)}")
print("Prototype only: validate false alarms on fresh readings before relying on it.")
