@echo off
setlocal
cd /d "%~dp0"

set "VENV_PY=%CD%\.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo [ERROR] .venv not found in %CD%.
    pause
    exit /b 1
)

if "%HF_HOME%"=="" if exist "A:\AI-Models\huggingface" set "HF_HOME=A:\AI-Models\huggingface"

echo Starting standalone Sanad QLoRA chat server on http://127.0.0.1:8002 ...
start "Open Sanad QLoRA Chat" /min cmd /c "for /L %%i in (1,1,90) do (timeout /t 2 /nobreak >nul & "%VENV_PY%" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8002/health', timeout=1.5)" >nul 2>&1 && (start "" "http://127.0.0.1:8002" & exit /b 0))"
"%VENV_PY%" scripts\serve_local.py --host 127.0.0.1 --port 8002
if errorlevel 1 (
    echo.
    echo [ERROR] Sanad QLoRA server exited with an error.
    pause
)
