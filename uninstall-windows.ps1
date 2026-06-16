# Uninstall LIS (Local Inference Server) — Windows.
# Removes the command(s): the OLD names (mlxs / mlx-launcher) AND the new one (lis-start),
# the pipx install, this repo's .venv, and the venv Scripts dir from your PATH. Your config
# (~/.config/mlx-launcher) is KEPT unless you pass -Purge.
#
#   powershell -ExecutionPolicy Bypass -File .\uninstall-windows.ps1
#   powershell -ExecutionPolicy Bypass -File .\uninstall-windows.ps1 -Purge
param([switch]$Purge)
$ErrorActionPreference = "SilentlyContinue"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $Here ".venv"
$Scripts = Join-Path $Venv "Scripts"

# 1) pipx install (distribution name is 'mlx-launcher' for both old + new command sets)
if (Get-Command pipx -ErrorAction SilentlyContinue) {
  pipx uninstall mlx-launcher | Out-Null
  Write-Host "checked pipx (removed mlx-launcher if it was installed)"
}

# 2) remove the venv Scripts dir from the user PATH (install-windows.ps1 added it)
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -and $userPath -like "*$Scripts*") {
  $new = ($userPath -split ';' | Where-Object { $_ -and $_ -ne $Scripts }) -join ';'
  [Environment]::SetEnvironmentVariable("Path", $new, "User")
  Write-Host "removed $Scripts from your user PATH"
}

# 3) this repo's .venv (holds lis-start.exe and the old mlxs/mlx-launcher shims)
if (Test-Path $Venv) { Remove-Item -Recurse -Force $Venv; Write-Host "removed $Venv" }

# 4) config / user data
$Cfg = Join-Path $env:USERPROFILE ".config\mlx-launcher"
if ($Purge) {
  if (Test-Path $Cfg) { Remove-Item -Recurse -Force $Cfg; Write-Host "removed $Cfg (profiles + chats)" }
} elseif (Test-Path $Cfg) {
  Write-Host "kept your config at $Cfg (use -Purge to delete profiles + chats)"
}

Write-Host ""
Write-Host "Done - LIS uninstalled. (llama.cpp is left installed - remove it separately if you want.)"
