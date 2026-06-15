# Start the simple local upload application.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Backend = "$Root\backend"
$Frontend = "$Root\frontend"
$Venv = "$Backend\.venv"

if (-not (Test-Path $Venv)) {
    python -m venv $Venv
}

& "$Venv\Scripts\pip.exe" install -q -r "$Backend\requirements.txt"

Write-Host "Starting API on http://localhost:8000" -ForegroundColor Green
$apiJob = Start-Job -ScriptBlock {
    param($Backend, $Venv)
    Set-Location $Backend
    $env:PYTHONPATH = $Backend
    & "$Venv\Scripts\uvicorn.exe" app.main:app --host 127.0.0.1 --port 8000 --reload
} -ArgumentList $Backend, $Venv

Start-Sleep -Seconds 2

$npmCmd = Get-Command npm -ErrorAction SilentlyContinue
if ($null -eq $npmCmd) {
    Write-Host "Node.js / npm is not installed or not on PATH." -ForegroundColor Yellow
    Write-Host "Backend is running at http://localhost:8000." -ForegroundColor Green
    Write-Host "Install Node.js and rerun .\scripts\start-local.ps1 to start the frontend UI." -ForegroundColor Yellow
    return
}

if (-not (Test-Path "$Frontend\node_modules")) {
    Set-Location $Frontend
    npm install
}

Write-Host "Open http://localhost:3003 and upload a dataset." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop." -ForegroundColor Gray

Set-Location $Frontend
$env:NEXT_PUBLIC_API_URL = "http://localhost:8000"
npm run dev
