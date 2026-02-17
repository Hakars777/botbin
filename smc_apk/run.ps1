if ($MyInvocation.InvocationName -ne '&') {
    powershell -ExecutionPolicy Bypass -File $MyInvocation.MyCommand.Definition
    exit
}

$ErrorActionPreference = 'Stop'
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $scriptDir

$pyCmd = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    $pyCmd = "py -3.12"
} else {
    $pyCmd = "python"
}

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment (Python 3.12)..."
    Invoke-Expression "$pyCmd -m venv .venv"
}

& ".venv\Scripts\Activate.ps1"

python -m pip install --upgrade pip 2>$null | Out-Null

if (Test-Path "requirements.txt") {
    Write-Host "Installing dependencies..."
    pip install -r requirements.txt
}

Write-Host "Starting SMC Alert..."
python main.py

pause
