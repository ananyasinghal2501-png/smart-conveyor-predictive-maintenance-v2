from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "main.py"
ML = ROOT / "ml"
BACKUP = ROOT / "main_before_anomaly_backup.py"

TRAIN = '''from pathlib import Path
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
'''

TEMPORAL = '''
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
'''

def once(s, old, new, label):
    if s.count(old) != 1:
        raise RuntimeError(f"Patch stopped safely: expected one '{label}' anchor, found {s.count(old)}.")
    return s.replace(old, new, 1)

def main():
    if not MAIN.exists() or not (ROOT / "ml" / "dataset.csv").exists():
        raise SystemExit("Place this script in your project root beside main.py; ml/dataset.csv must exist.")
    ML.mkdir(exist_ok=True)
    (ML / "train_anomaly_model.py").write_text(TRAIN, encoding="utf-8")
    s = MAIN.read_text(encoding="utf-8")
    if "TEMPORAL ANALYSIS ADD-ON" in s:
        print("Anomaly support already patched. Training script is ready.")
        return
    if not BACKUP.exists():
        shutil.copy2(MAIN, BACKUP)

    s = once(s, "from datetime import datetime",
             "from datetime import datetime\nfrom collections import deque\nfrom statistics import median",
             "imports")
    s = once(s, '    print("WARNING: conveyor_model.pkl not found")',
             '    print("WARNING: conveyor_model.pkl not found")\n\nANOMALY_MODEL_PATH = Path(__file__).parent / "ml" / "anomaly_model.pkl"\nanomaly_model = joblib.load(ANOMALY_MODEL_PATH) if ANOMALY_MODEL_PATH.exists() else None\nif anomaly_model is None:\n    print("Anomaly model not found yet; train ml/train_anomaly_model.py")',
             "model loader")
    s = once(s, "sensor_history = []",
             "sensor_history = []\n\n# TEMPORAL ANALYSIS ADD-ON\n" + TEMPORAL,
             "temporal helpers")

    analysis = '''
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
'''
    s = once(s, '    timestamp = datetime.now().isoformat()',
             '    timestamp = datetime.now().isoformat()\n' + analysis,
             "request analysis")
    s = once(s, "    # DYNAMIC HEALTH SCORE",
             '    if data.flame != 0 and anomaly_detected and severity == "Normal":\n        severity = "Warning"\n\n    # DYNAMIC HEALTH SCORE',
             "advisory severity")
    s = once(s, "    # UPDATE CURRENT STATE",
             '    if anomaly_detected:\n        why_status.append("ANOMALY WARNING: unusual sensor behaviour; investigate. This is not a confirmed fault.")\n\n    # UPDATE CURRENT STATE',
             "status explanation")
    health_anchor = '        "confidence":\n            current_state["confidence"],'
    health_replacement = (
        '        "confidence":\n            current_state["confidence"],\n'
        '        "anomaly_detected": current_state["anomaly_detected"],\n'
        '        "anomaly_status": current_state["anomaly_status"],\n'
        '        "anomaly_score": current_state["anomaly_score"],\n'
        '        "temporal_analysis": current_state["temporal_analysis"],'
    )
    if health_anchor not in s:
        raise RuntimeError("Patch stopped safely: health endpoint anchor not found.")
    # /health appears before /predictions, so replace only its first occurrence.
    s = s.replace(health_anchor, health_replacement, 1)
    MAIN.write_text(s, encoding="utf-8")
    print("Done. main.py updated; backup saved as main_before_anomaly_backup.py.")
    print("Next: python ml/train_anomaly_model.py")
    print("Then restart FastAPI. Anomaly flags are advisory and never stop the motor.")

if __name__ == "__main__":
    main()
