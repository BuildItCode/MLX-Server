# Install the launcher on Windows with the llama.cpp backend.
# MLX is Apple-Silicon-only, so on Windows the usable engine is llama.cpp (llama-server.exe).
# This installs the launcher (command: lis-start) and best-effort fetches a prebuilt llama-server.exe.
#
#   powershell -ExecutionPolicy Bypass -File .\install-windows.ps1
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Here

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

$Py = Find-Python
if (-not $Py) {
  Write-Error "No suitable Python found (need 3.10-3.14). Install it: 'winget install Python.Python.3.12' or python.org."
  exit 1
}
Write-Host "Using Python: $Py"
$pyParts = $Py.Split(" ")

# --- 1) the launcher (pipx if available, else venv + PATH) -------------------------
$installed = $false
if (Get-Command pipx -ErrorAction SilentlyContinue) {
  Write-Host "Installing the launcher with pipx ..."
  & pipx install --force $Here
  if ($LASTEXITCODE -eq 0) { & pipx ensurepath | Out-Null; $installed = $true }
}
if (-not $installed) {
  $Venv = Join-Path $Here ".venv"
  if (-not (Test-Path $Venv)) { & $pyParts[0] @($pyParts[1..($pyParts.Length - 1)]) -m venv $Venv }
  $vpy = Join-Path $Venv "Scripts\python.exe"
  & $vpy -m pip install --quiet --upgrade pip
  & $vpy -m pip install --quiet -e $Here
  $scripts = Join-Path $Venv "Scripts"
  $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
  if ($userPath -notlike "*$scripts*") {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$scripts", "User")
    Write-Host "  added $scripts to your user PATH (restart the terminal)."
  }
}

# --- 2) llama.cpp (llama-server.exe) -----------------------------------------------
if (Get-Command llama-server -ErrorAction SilentlyContinue) {
  Write-Host "OK: llama-server already on PATH."
} else {
  Write-Host "Fetching a prebuilt llama-server from github.com/ggml-org/llama.cpp ..."
  $arch = if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") { "win-cpu-arm64" } else { "win-cpu-x64" }
  try {
    $rel = Invoke-RestMethod "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
    $asset = $rel.assets | Where-Object { $_.name -like "*bin-$arch.zip" } | Select-Object -First 1
    if (-not $asset) { throw "no matching release asset for $arch" }
    $dest = Join-Path $env:LOCALAPPDATA "llama.cpp"
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    $zip = Join-Path $env:TEMP $asset.name
    Invoke-WebRequest $asset.browser_download_url -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath $dest -Force
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($userPath -notlike "*$dest*") {
      [Environment]::SetEnvironmentVariable("Path", "$userPath;$dest", "User")
    }
    Write-Host "OK: installed llama-server -> $dest (CPU build; added to PATH, restart the terminal)."
  } catch {
    Write-Host "Couldn't auto-install llama-server. Install it manually:"
    Write-Host "  - download a build from https://github.com/ggml-org/llama.cpp/releases"
    Write-Host "      win-cpu-x64 = CPU · win-vulkan/cuda = GPU"
    Write-Host "  - or search winget: 'winget search llama'"
    Write-Host "  Put llama-server.exe on your PATH."
  }
}

Write-Host ""
Write-Host "Done. Start it from a new terminal with:  lis-start"
