@echo off
rem Possession Viewer. Usage: run_viewer.bat [data-dir] [labels.csv]
rem Defaults: MTAH_DATA_DIR (or E:\barca) and submissions\EUCLIDES.csv
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
set "DATA=%~1"
if "%DATA%"=="" set "DATA=%MTAH_DATA_DIR%"
if "%DATA%"=="" set "DATA=E:\barca"
set "LABELS=%~2"
if "%LABELS%"=="" set "LABELS=submissions\EUCLIDES.csv"
python -m viewer --data-dir "%DATA%" --labels "%LABELS%"
endlocal
