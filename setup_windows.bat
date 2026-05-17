@echo off
echo ==========================================
echo   MAPE-K Demo - Windows Setup
echo ==========================================
echo.
echo This will create a Python virtual environment
echo and install all required packages.
echo.
pause

cd /d "%~dp0"

echo [1/3] Creating virtual environment...
python -m venv venv_win
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Install Python 3.9+ from python.org
    pause
    exit /b 1
)

echo [2/3] Installing packages...
call venv_win\Scripts\activate.bat

pip install flask flask-cors pandas numpy scikit-learn scipy streamlit requests

echo [3/3] Checking installation...
python -c "import flask, pandas, numpy, sklearn, streamlit; print('All packages installed successfully!')"

echo.
echo ==========================================
echo   Setup Complete!
echo ==========================================
echo.
echo To start the demo, run: start_full_demo.bat
echo.
pause
