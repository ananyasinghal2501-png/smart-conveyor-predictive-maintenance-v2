import serial
import re
import json
import time
import urllib.request

PORT = "COM13"
BAUD_RATE = 115200
API_URL = "http://127.0.0.1:8000/sensor-data"

pattern = re.compile(
    r"Mag:\s*(-?\d+(?:\.\d+)?)\s+"
    r"Dev:\s*(-?\d+(?:\.\d+)?)\s+"
    r"Obj:\s*(-?\d+(?:\.\d+)?)C\s+"
    r"Amb:\s*(-?\d+(?:\.\d+)?)\s+"
    r"Load:\s*(-?\d+(?:\.\d+)?)g\s+"
    r"Flame:\s*(\d+)\s+"
    r"Speed:\s*(-?\d+(?:\.\d+)?)"
)

print("Opening ESP32 on", PORT)

ser = serial.Serial(
    PORT,
    BAUD_RATE,
    timeout=1
)

time.sleep(2)

print("ESP32 connected!")
print("Reading live sensor data...")

last_send = 0
last_payload = None


def send_to_fastapi(payload):

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        API_URL,
        data=data,
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=2
        ) as response:

            print(
                ">>> Sent to FastAPI:",
                response.status,
                payload
            )

    except Exception as error:

        print(
            ">>> FastAPI error:",
            error
        )


while True:

    try:

        line = (
            ser.readline()
            .decode("utf-8", errors="ignore")
            .strip()
        )

        if not line:
            continue

        print("ESP32:", line)


        # =====================================
        # IMMEDIATE FLAME EMERGENCY
        # =====================================

        if "FLAME DETECTED - BELT STOPPED" in line:

            print(">>> EMERGENCY FLAME EVENT RECEIVED")

            if last_payload is not None:

                emergency_payload = last_payload.copy()

                emergency_payload["flame"] = 0
                emergency_payload["speed"] = 0

                send_to_fastapi(
                    emergency_payload
                )

            continue


        # =====================================
        # FLAME CLEARED
        # =====================================

        if "Flame cleared - belt resumed" in line:

            print(">>> FLAME CLEARED")

            if last_payload is not None:

                clear_payload = last_payload.copy()

                clear_payload["flame"] = 1
                clear_payload["speed"] = 150

                send_to_fastapi(
                    clear_payload
                )

            continue


        # =====================================
        # NORMAL SENSOR LINE
        # =====================================

        match = pattern.search(line)

        if not match:
            continue


        flame_now = int(
            match.group(6)
        )


        payload = {

            "vibration":
                float(match.group(1)),

            "alignment":
                float(match.group(2)),

            "object_temperature":
                float(match.group(3)),

            "ambient_temperature":
                float(match.group(4)),

            "load":
                float(match.group(5)),

            "flame":
                flame_now,

            "speed":
                float(match.group(7))
        }


        # Remember newest actual sensor values
        last_payload = payload.copy()


        # Flame packets are NEVER delayed
        if flame_now == 0:

            print(
                ">>> FLAME = 0 DETECTED"
            )

            send_to_fastapi(
                payload
            )

            last_send = time.time()

            continue


        # Normal readings about twice per second
        if time.time() - last_send < 0.5:
            continue


        send_to_fastapi(
            payload
        )

        last_send = time.time()


    except KeyboardInterrupt:

        print(
            "\nStopping serial bridge..."
        )

        ser.close()
        break


    except Exception as error:

        print(
            "Serial error:",
            error
        )

        time.sleep(1)