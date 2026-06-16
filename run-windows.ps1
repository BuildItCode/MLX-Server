# Bootstrap (create venv + install) and launch the launcher on Windows (llama.cpp backend).
#   powershell -ExecutionPolicy Bypass -File .\run-windows.ps1
#   powershell -ExecutionPolicy Bypass -File .\run-windows.ps1 -Reinstall
param([switch]$Reinstall)
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Here
$Venv = Join-Path $Here ".venv"

function Find-Python {
  foreach ($cmd in @("py -3.13", "py -3.12", "py -3.11", "py -3.10", "py -3", "python")) {
    $parts = $cmd.Split(" ")
    if (Get-Command $parts[0] -ErrorAction SilentlyContinue) {
      & $parts[0] @($parts[1..($parts.Length - 1)]) -c "import sys; sys.exit(0 if (3,10)<=sys.version_info<(3,15) else 1)" 2>$null
      if ($LASTEXITCODE -eq 0) { return $cmd }
    }
  }
  return $null
}

if (-not (Test-Path $Venv)) {
  $Py = Find-Python
  if (-not $Py) { Write-Error "No suitable Python found (need 3.10-3.14). Install from python.org."; exit 1 }
  $parts = $Py.Split(" ")
  Write-Host "Creating virtual environment in .venv ..."
  & $parts[0] @($parts[1..($parts.Length - 1)]) -m venv $Venv
}

$vpy = Join-Path $Venv "Scripts\python.exe"
$lis = Join-Path $Venv "Scripts\lis-start.exe"
if ($Reinstall -or -not (Test-Path $lis)) {
  Write-Host "Installing dependencies (this runs only when needed) ..."
  & $vpy -m pip install --quiet --upgrade pip
  & $vpy -m pip install --quiet -e $Here
}

if (-not (Get-Command llama-server -ErrorAction SilentlyContinue)) {
  Write-Host "NOTE: llama-server not found - run .\install-windows.ps1 or install llama.cpp first."
}

& $lis @args
