"""
Fake ESP32 - sends sensor readings to FastAPI so you can test the dashboard
and the 3D twin without hardware.

Usage (backend must be running on port 8000):
    python simulate.py normal      # healthy belt
    python simulate.py warning     # temp + vibration a bit high
    python simulate.py critical    # overheating, heavy vibration, misalignment
    python simulate.py flame       # flame detected, motor stopped
    python simulate.py loop        # normal -> warning -> critical -> flame -> repeat
"""
import json
import random
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8000/sensor-data"

SCENARIOS = {
    #            obj_temp amb  vib   align load speed flame
    "normal":   (32.0,    28.0, 0.8,  0.10, 120.0, 200, 1),
    "warning":  (74.0,    29.0, 5.5,  0.60, 120.0, 200, 1),
    "critical": (86.0,    30.0, 9.0,  1.20, 120.0, 200, 1),
    "flame":    (95.0,    31.0, 1.0,  0.10, 120.0, 0,   0),
}


def build(name):
    t, a, v, al, ld, sp, fl = SCENARIOS[name]
    j = lambda x, s: round(x + random.uniform(-s, s), 2)   # small noise
    return {
        "object_temperature": j(t, 0.4),
        "ambient_temperature": j(a, 0.2),
        "vibration": max(0.0, j(v, 0.1)),
        "alignment": j(al, 0.02),
        "load": j(ld, 2.0),
        "speed": sp,
        "flame": fl,
    }


def send(payload):
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def run(name, seconds):
    end = time.time() + seconds
    while time.time() < end:
        state = send(build(name))
        print(f"{name:9s} -> severity={state['severity']:8s} "
              f"parts={[p['part'] for p in state['fault_parts']]}")
        time.sleep(0.5)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "normal"
    if mode == "loop":
        while True:
            for n in ("normal", "warning", "critical", "flame"):
                run(n, 8)
    elif mode in SCENARIOS:
        run(mode, 10)
    else:
        sys.exit(__doc__)
