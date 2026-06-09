[CmdletBinding()]
param(
    [string]$RequirementsFile = "requirements.txt",
    [string]$VenvDir = ".venv",
    [string]$PythonExecutable = "python",
    [switch]$UpgradePip
)

$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
if (-not $ProjectRoot) {
    $ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
}

Set-Location $ProjectRoot

$RequirementsPath = Join-Path $ProjectRoot $RequirementsFile
$VenvPath = Join-Path $ProjectRoot $VenvDir
$PythonExe = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Get-Command $PythonExecutable -ErrorAction SilentlyContinue)) {
    throw "Python executable not found: $PythonExecutable"
}

if (-not (Test-Path $RequirementsPath)) {
    throw "Requirements file not found: $RequirementsPath"
}

if (-not (Test-Path $PythonExe)) {
    Write-Host "Creating virtual environment at $VenvPath ..."
    & $PythonExecutable -m venv $VenvPath
}
else {
    Write-Host "Reusing existing virtual environment at $VenvPath ..."
}

if ($UpgradePip) {
    Write-Host "Upgrading pip ..."
    & $PythonExe -m pip install --upgrade pip
}

Write-Host "Installing packages from $RequirementsFile ..."
& $PythonExe -m pip install -r $RequirementsPath

Write-Host ""
Write-Host "Setup complete."
Write-Host "Activate this environment in the current terminal with:"
Write-Host (".\{0}\Scripts\Activate.ps1" -f $VenvDir)
Write-Host ""
Write-Host "Interpreter path:"
Write-Host $PythonExe