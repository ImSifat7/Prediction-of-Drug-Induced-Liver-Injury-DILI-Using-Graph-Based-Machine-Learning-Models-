@echo off
REM Recreate the virtual environment cleanly with the active Python.
REM The current venv was built for Python 3.13 which is no longer installed,
REM so its python.exe shim is broken. This script rebuilds it.

set PROJ=%~dp0
cd /d "%PROJ%"

echo [setup] Removing old venv...
if exist venv rmdir /s /q venv

echo [setup] Creating fresh venv...
py -m venv venv
if errorlevel 1 (
    echo [setup] Failed to create venv. Make sure Python 3.10+ is on PATH.
    exit /b 1
)

call venv\Scripts\activate.bat

echo [setup] Upgrading pip...
python -m pip install --upgrade pip

echo [setup] Installing dependencies...
pip install numpy pandas scikit-learn scipy matplotlib optuna rdkit
pip install torch
pip install torch_geometric

echo [setup] Done. Activate with:
echo    venv\Scripts\activate
