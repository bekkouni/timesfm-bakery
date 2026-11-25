@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Create the project virtual environment first. See README.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py --server.headless false --browser.serverAddress localhost
if errorlevel 1 pause
