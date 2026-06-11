# Start the local development stack on Windows
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ScriptPath = Join-Path $Root "scripts/dev-up.ps1"

if (-not (Test-Path $ScriptPath)) {
    throw "Could not find scripts/dev-up.ps1"
}

& $ScriptPath
