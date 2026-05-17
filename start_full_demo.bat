@echo off
setlocal enabledelayedexpansion

echo.
echo ==========================================
echo   MAPE-K Agentic Fraud Detection Demo
echo ==========================================
echo.

cd /d "%~dp0"

REM Check if venv exists
if not exist "venv_win\Scripts\python.exe" (
    if exist "venv_new\Scripts\python.exe" (
        set VENV_DIR=venv_new
    ) else (
        echo ERROR: Virtual environment not found!
        echo Run setup_windows.bat first to install dependencies.
        echo.
        pause
        exit /b 1
    )
) else (
    set VENV_DIR=venv_win
)

echo [1/4] Cleaning up existing processes on port 8000 and 8501...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo   Killing process on port 8000 (PID: %%a)
    taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8501" ^| findstr "LISTENING"') do (
    echo   Killing process on port 8501 (PID: %%a)
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 2 /nobreak >nul

echo [2/4] Starting API server - pipeline will train automatically...
echo   (A new window will open - DO NOT CLOSE IT)
echo.

set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
start "MAPE-K API" cmd /c "title MAPE-K API && set OMP_NUM_THREADS=1 && set MKL_NUM_THREADS=1 && set OPENBLAS_NUM_THREADS=1 && %VENV_DIR%\Scripts\python.exe simple_api.py && echo. && echo Server stopped. Press any key to close... && pause"

echo [3/4] Waiting for pipeline training...
echo.
timeout /t 10 /nobreak >nul
echo   API should be ready!

:api_ready
echo.
echo [4/4] Starting Streamlit UI...
start "Streamlit UI" cmd /c "title Streamlit UI && streamlit run app/frontend/streamlit_app.py --server.port 8501 --server.headless true"

timeout /t 3 /nobreak >nul

echo.
echo ==========================================
echo   DEMO READY
echo ==========================================
echo.
echo   Streamlit UI:  http://localhost:8501
echo   API Server:    http://localhost:8000
echo   Live Metrics:  http://localhost:8000/api/metrics
echo   Drift Status:  http://localhost:8000/api/drift
echo.
echo   Demo examples: type demo_examples.txt
echo.
echo   To stop: Close both terminal windows
echo ==========================================
echo.
pause
