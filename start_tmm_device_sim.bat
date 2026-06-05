@echo off
setlocal

cd /d "%~dp0"
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"

where pythonw >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    start "" pythonw -m tmm_device_sim
    exit /b 0
)

where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    start "" python -m tmm_device_sim
    exit /b 0
)

where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    start "" py -3 -m tmm_device_sim
    exit /b 0
)

echo Could not find Python. Please install Python and run:
echo python -m pip install -r requirements.txt
pause
exit /b 1
