@echo off
rem Possession Viewer. Usage: run_viewer.bat [data-dir] [labels.csv] [port]
rem Defaults: MTAH_DATA_DIR (or E:\barca), submissions\EUCLIDES.csv, port 8000 (or VIEWER_PORT)
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
set "DATA=%~1"
if "%DATA%"=="" set "DATA=%MTAH_DATA_DIR%"
if "%DATA%"=="" set "DATA=E:\barca"
set "LABELS=%~2"
if "%LABELS%"=="" set "LABELS=submissions\EUCLIDES.csv"
set "PORT=%~3"
if "%PORT%"=="" set "PORT=%VIEWER_PORT%"
if "%PORT%"=="" set "PORT=8000"
python -m viewer --data-dir "%DATA%" --labels "%LABELS%" --port %PORT%
endlocal
