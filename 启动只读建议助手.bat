@echo off
chcp 65001 >nul
setlocal
set "ENTRY=%~dp0scripts\pvz_readonly_advisor.py"

if not exist "%ENTRY%" (
    echo [PVZagent] Entry script not found: %ENTRY%
    pause
    exit /b 1
)

set "PYW="
if exist "%~dp0.venv\Scripts\pythonw.exe" (
    set "PYW=%~dp0.venv\Scripts\pythonw.exe"
)
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
if not defined PYW if exist "D:\Python\pythonw.exe" (
    set "PYW=D:\Python\pythonw.exe"
)

if not defined PYW (
    echo [PVZagent] 未找到 pythonw.exe，请先安装 Python 3.11+ 并运行 pip install -r requirements.txt
    pause
    exit /b 1
)

start "" /D "%~dp0" "%PYW%" "%ENTRY%"
exit /b 0
