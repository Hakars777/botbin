@echo off
cd /d "%~dp0"

where py >nul 2>&1
if %errorlevel%==0 (
    set PYTHON=py -3.12
) else (
    set PYTHON=python
)

if not exist ".venv" (
    echo Creating virtual environment (Python 3.12)...
    %PYTHON% -m venv .venv
)

call .venv\Scripts\activate.bat

python -m pip install --upgrade pip >nul 2>&1

if exist requirements.txt (
    echo Installing dependencies...
    pip install -r requirements.txt
)

echo Starting SMC Alert...
python main.py

pause
