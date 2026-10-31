SMART CONVEYOR CAMERA VISION INTEGRATION

FILES
- main.py: adds /camera-data and /camera-status. Camera state is separate from ESP32 sensor state.
- index.html: adds a Camera Vision panel and handles camera-only WebSocket updates.
- belt_vision.py: OpenCV prototype that reads the ESP32-CAM stream and posts camera measurements.

BEFORE REPLACING FILES
1. In your project folder, make backups of the current main.py and index.html.
2. Replace those two files with the versions in this package. Keep database.py unchanged.
3. Save belt_vision.py in the project folder on the laptop that will run OpenCV.

NETWORK SETTINGS TO VERIFY
- Camera stream defaults to http://192.168.4.50:81/stream
- Backend endpoint defaults to http://192.168.4.2:8000/camera-data
If ipconfig on the backend laptop shows a different ConveyorRig IPv4 address, edit BACKEND_URL in belt_vision.py to use that address. If the camera IP differs, update STREAM_URL in belt_vision.py and data-src in index.html.

RUN
1. On the backend laptop, start the API from the project folder:
   python -m uvicorn main:app --host 0.0.0.0 --port 8000
2. On the backend laptop, open index.html with Live Server, normally http://127.0.0.1:5500/index.html.
3. On the laptop that can access the camera, install dependencies if needed:
   python -m pip install opencv-python numpy requests
4. From the project folder, run:
   python belt_vision.py
5. In the OpenCV preview, press C when the belt is centred to set the reference. Move the belt edge laterally (or test with a dark strip against a light background) and watch CAMERA ALIGNMENT on the dashboard change. Press R to reset the possible-tear flag and Q to quit.
6. Optional endpoint check from a browser on the rig network:
   http://192.168.4.2:8000/camera-status

IMPORTANT LIMITS
- Camera alignment is measured in pixels and does not replace the ESP32 alignment field used by the ML model.
- Tear detection is a basic dark-belt/white-gap heuristic, not a validated defect classifier. Lighting, camera angle, belt colour, shadows, and background can affect results. Treat alerts as candidates for human inspection; this prototype does not stop the conveyor or control safety hardware.
- The dashboard live stream and OpenCV script may both connect to the camera. If the stream becomes unstable, test with only one viewer at a time or temporarily disable the dashboard stream.
