@echo off
REM Meter Reading API - Startup Script
REM Starts the FastAPI server

echo ========================================
echo Meter Reading API - Starting Server
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo [!] No virtual environment found
    echo [*] Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies if needed
if not exist "venv\Lib\site-packages\fastapi" (
    echo [*] Installing dependencies...
    pip install -r requirements.txt
)

echo.
echo [*] Starting FastAPI server...
echo.
echo Server will be available at:
echo   - http://localhost:8000
echo   - http://localhost:8000/docs (Swagger UI)
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start server
python main.py
