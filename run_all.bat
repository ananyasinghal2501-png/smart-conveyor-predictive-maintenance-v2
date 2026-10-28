
@echo off
cd /d "%~dp0"

set "PY=%CD%\.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo ERROR: Python virtual environment not found.
    pause
    exit /b 1
)

echo ==========================================
echo SMART CONVEYOR PREDICTIVE MAINTENANCE
echo Starting project...
echo ==========================================

start "FastAPI Backend" cmd /k ""%PY%" -m uvicorn main:app --host 0.0.0.0 --port 8000"

timeout /t 3 /nobreak >nul

start "Dashboard Server" cmd /k ""%PY%" -m http.server 5500"

timeout /t 2 /nobreak >nul

start "Digital Twin Simulator" cmd /k ""%PY%" simulate.py"

timeout /t 2 /nobreak >nul

start "" "http://127.0.0.1:5500/index.html"

echo.
echo Project launch commands completed.
echo Keep the terminal windows open.
echo Camera vision requires the camera stream to be reachable.
pause

