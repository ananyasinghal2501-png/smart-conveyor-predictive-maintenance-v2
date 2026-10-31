"""ESP32-CAM belt vision prototype.

Run this on the laptop that can reach the ESP32-CAM stream. It sends camera-only
measurements to the FastAPI backend and deliberately does NOT overwrite the
ESP32 sensor's alignment value used by the ML model.

Controls in the preview window:
  c = set the current belt centre as the alignment reference
  r = reset the latched visual tear flag
  q = quit

The tear detector is a simple contrast/gap heuristic for prototyping, not a
validated industrial defect detector. Test under controlled lighting.
"""
import os
import threading
import time

import cv2
import numpy as np
import requests

# Update these if your network IPs are different.
STREAM_URL = "http://192.168.4.50:81/stream"
BACKEND_URL = "http://192.168.4.2:8000/camera-data"
POST_EVERY_S = 0.5

# Tune these only after observing your actual camera view.
TEAR_RATIO = 0.04
TEAR_FRAMES = 5
SMOOTH_N = 5
SNAP_DIR = "snapshots"


class FrameGrabber:
    """Read the MJPEG stream in a background thread."""
    def __init__(self, url):
        self.url = url
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.connected = False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.running:
            cap = cv2.VideoCapture(self.url)
            if not cap.isOpened():
                self.connected = False
                cap.release()
                time.sleep(2)
                continue
            self.connected = True
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    break
                with self.lock:
                    self.frame = frame
            cap.release()
            self.connected = False
            time.sleep(1)

    def get(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False


def analyse(frame):
    """Estimate the largest dark belt region and white gaps within its hull."""
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, dark = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    belt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(belt) < 0.10 * width * height:
        return None

    belt_fill = np.zeros((height, width), np.uint8)
    cv2.drawContours(belt_fill, [belt], -1, 255, -1)
    hull = cv2.convexHull(belt)
    hull_mask = np.zeros((height, width), np.uint8)
    cv2.drawContours(hull_mask, [hull], -1, 255, -1)
    hull_area = max(cv2.countNonZero(hull_mask), 1)
    dark_in_hull = cv2.countNonZero(cv2.bitwise_and(dark, hull_mask))
    tear_ratio = max(0.0, (hull_area - dark_in_hull) / hull_area)

    lefts, rights = [], []
    for y in range(int(height * 0.30), int(height * 0.70), 4):
        xs = np.where(belt_fill[y] > 0)[0]
        if xs.size:
            lefts.append(int(xs.min()))
            rights.append(int(xs.max()))
    if not lefts:
        return None

    left = float(np.median(lefts))
    right = float(np.median(rights))
    return {
        "left": left,
        "right": right,
        "center": (left + right) / 2.0,
        "tear_ratio": tear_ratio,
        "hull": hull,
        "dark": dark,
    }


def post_async(state, payload):
    """Post without blocking the video preview loop."""
    if state["posting"]:
        return
    state["posting"] = True

    def send():
        try:
            response = requests.post(BACKEND_URL, json=payload, timeout=1.5)
            response.raise_for_status()
            state["post_ok"] = True
        except Exception as exc:
            state["post_ok"] = False
            state["last_error"] = str(exc)
        finally:
            state["posting"] = False

    threading.Thread(target=send, daemon=True).start()


def main():
    os.makedirs(SNAP_DIR, exist_ok=True)
    grabber = FrameGrabber(STREAM_URL)
    reference_center = None
    history = []
    tear_count = 0
    tear_latched = False
    last_post = 0.0
    fps = 0.0
    last_frame_time = time.time()
    state = {"posting": False, "post_ok": None, "last_error": ""}

    print("Camera stream:", STREAM_URL)
    print("Posting camera data to:", BACKEND_URL)
    print("Keys: c = set reference | r = reset tear flag | q = quit")
    print("Note: this is a visual prototype; validate the camera view before relying on alerts.")

    try:
        while True:
            frame = grabber.get()
            if frame is None:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                time.sleep(0.05)
                continue

            now = time.time()
            instantaneous_fps = 1.0 / max(now - last_frame_time, 1e-3)
            fps = instantaneous_fps if fps == 0 else (0.9 * fps + 0.1 * instantaneous_fps)
            last_frame_time = now

            result = analyse(frame)
            shown = frame.copy()
            height, width = shown.shape[:2]
            belt_found = result is not None
            alignment_px = 0.0
            tear_ratio = 0.0

            if belt_found:
                tear_ratio = float(result["tear_ratio"])
                offset = 0.0 if reference_center is None else result["center"] - reference_center
                history.append(offset)
                history = history[-SMOOTH_N:]
                alignment_px = round(sum(history) / len(history), 1)

                if tear_ratio > TEAR_RATIO:
                    tear_count += 1
                else:
                    tear_count = 0
                if tear_count >= TEAR_FRAMES and not tear_latched:
                    tear_latched = True
                    snapshot = os.path.join(SNAP_DIR, "possible_tear_%d.jpg" % int(time.time()))
                    cv2.imwrite(snapshot, frame)
                    print("POSSIBLE TEAR FLAGGED — snapshot saved:", snapshot)

                cv2.drawContours(shown, [result["hull"]], -1, (0, 255, 255), 1)
                cv2.line(shown, (int(result["left"]), 0), (int(result["left"]), height), (0, 255, 0), 1)
                cv2.line(shown, (int(result["right"]), 0), (int(result["right"]), height), (0, 255, 0), 1)
                cv2.line(shown, (int(result["center"]), 0), (int(result["center"]), height), (255, 0, 0), 1)
                if reference_center is not None:
                    cv2.line(shown, (int(reference_center), 0), (int(reference_center), height), (0, 0, 255), 1)
                cv2.imshow("belt mask (prototype)", result["dark"])
            else:
                tear_count = 0

            color = (0, 0, 255) if tear_latched else (0, 200, 0)
            cv2.putText(shown, "belt: %s" % ("FOUND" if belt_found else "NOT FOUND"), (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            ref_label = "reference set" if reference_center is not None else "press C to set reference"
            cv2.putText(shown, "alignment: %.1f px (%s)" % (alignment_px, ref_label), (5, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
            cv2.putText(shown, "gap ratio: %.4f | possible tear: %s" % (tear_ratio, "YES" if tear_latched else "no"), (5, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            post_text = "OK" if state["post_ok"] is True else "FAILED" if state["post_ok"] is False else "WAITING"
            cv2.putText(shown, "fps: %.1f | API post: %s" % (fps, post_text), (5, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            cv2.imshow("ESP32-CAM belt vision", shown)

            if now - last_post >= POST_EVERY_S:
                last_post = now
                post_async(state, {
                    "alignment_px": alignment_px,
                    "tear_detected": tear_latched,
                    "tear_ratio": round(tear_ratio, 4),
                    "belt_found": belt_found,
                    "camera_fps": round(fps, 1),
                    "reference_set": reference_center is not None,
                })

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c") and belt_found:
                reference_center = result["center"]
                history = []
                print("Alignment reference set at x = %.1f px" % reference_center)
            elif key == ord("r"):
                tear_latched = False
                tear_count = 0
                print("Visual tear flag reset")

    finally:
        grabber.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
