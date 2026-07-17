@echo off
setlocal

echo ============================================================
echo  Drone Detection - Field Laptop Setup
echo ============================================================
echo.

:: Detect Python 3.13 via the launcher (works regardless of username or admin elevation)
py -3.13 --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.13 not found. Install from https://www.python.org/downloads/ and re-run.
    pause
    exit /b 1
)
for /f "delims=" %%P in ('py -3.13 -c "import sys; print(sys.executable)"') do set PYTHON=%%P
echo Found Python: %PYTHON%
echo.

:: Always recreate venv to ensure it points to the correct Python
echo Creating virtual environment ^(.venv^)...
if exist ".venv" (
    echo   Removing old .venv...
    rmdir /s /q .venv
)
"%PYTHON%" -m venv .venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)
echo   .venv created.
echo.

:: Upgrade pip
echo Upgrading pip...
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
echo.

:: Install PyTorch with CUDA 12.8
echo Installing PyTorch with CUDA 12.8 (large download ~2.7GB, be patient)...
.venv\Scripts\pip.exe install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --quiet
if errorlevel 1 (
    echo [ERROR] PyTorch install failed.
    pause
    exit /b 1
)
echo   PyTorch installed OK.
echo.

:: Install remaining dependencies
echo Installing remaining dependencies...
.venv\Scripts\pip.exe install librosa scipy numpy sounddevice pyserial pynmea2 --quiet
if errorlevel 1 (
    echo [ERROR] Dependency install failed.
    pause
    exit /b 1
)
echo   Dependencies installed OK.
echo.

:: Quick sanity check
echo Running pipeline sanity check...
.venv\Scripts\python.exe -c "import torch; import librosa; import sounddevice; print('  torch:', torch.__version__, '| CUDA:', torch.cuda.is_available()); print('  librosa OK | sounddevice OK')"
if errorlevel 1 (
    echo [WARNING] Sanity check failed - check install above.
) else (
    echo.
    echo ============================================================
    echo  Setup complete. To run:
    echo    .venv\Scripts\python.exe DroneDetectorGUI.py
    echo    .venv\Scripts\python.exe RealTimeBearingGPS.py
    echo ============================================================
)
echo.
pause
