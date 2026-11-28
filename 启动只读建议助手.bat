@echo off
setlocal
set "ENTRY=%~dp0scripts\pvz_readonly_advisor.py"
if not exist "%ENTRY%" (
    echo [PVZagent] Entry script not found: %ENTRY%
    pause
    exit /b 1
)
set "PYW="
if defined PVZ_PYTHONW if exist "%PVZ_PYTHONW%" set "PYW=%PVZ_PYTHONW%"
if not defined PYW if exist "%~dp0.venv\Scripts\pythonw.exe" set "PYW=%~dp0.venv\Scripts\pythonw.exe"
if not defined PYW (
    for /f "delims=" %%I in ('where pythonw.exe 2^>nul') do (
        if not defined PYW set "PYW=%%I"
    )
)
if not defined PYW (
    for /f "delims=" %%I in ('where pyw.exe 2^>nul') do (
        if not defined PYW set "PYW=%%I"
    )
)
if not defined PYW (
    echo [PVZagent] pythonw.exe not found. Install Python 3.11+ or set PVZ_PYTHONW, then run: pip install -r requirements.txt
    pause
    exit /b 1
)
start "" /D "%~dp0" "%PYW%" "%ENTRY%"
exit /b 0
