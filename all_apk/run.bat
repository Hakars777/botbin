@echo off
cd /d "%~dp0"
if not exist ".venv" (
    echo Creating virtual environment...
    py -3.12 -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul 2>&1
if exist requirements.txt (
    pip install -r requirements.txt >nul 2>&1
)
python main.py
pause
