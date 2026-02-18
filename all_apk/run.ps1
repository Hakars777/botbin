Set-Location $PSScriptRoot
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    py -3.12 -m venv .venv
}
& ".venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip 2>&1 | Out-Null
if (Test-Path "requirements.txt") {
    pip install -r requirements.txt 2>&1 | Out-Null
}
python main.py
pause
