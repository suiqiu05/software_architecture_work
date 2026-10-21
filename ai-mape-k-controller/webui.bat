@echo off
rem ============================================================
rem  webui.bat - start the browser console (portable)
rem
rem  With no arguments it starts on http://127.0.0.1:8765 and
rem  opens your browser automatically.
rem  Extra arguments are forwarded, e.g.:  webui.bat --port 9000
rem  Press Ctrl+C in this window to stop the server.
rem ============================================================
setlocal
cd /d "%~dp0"

set "VPY=%~dp0.venv\Scripts\python.exe"
if exist "%VPY%" goto :found

set "VPY="
call :probe "py -3"
if not defined VPY call :probe "python"
if not defined VPY call :probe "python3"
if defined VPY goto :found

echo.
echo [ERROR] No usable Python environment found.
echo         Please run setup_env.bat first.
pause
exit /b 1

:probe
for /f "delims=" %%i in ('%~1 -c "import numpy,sys; print(sys.executable)" 2^>nul') do (
  if not defined VPY set "VPY=%%i"
)
exit /b 0

:found
if "%~1"=="" (
  set "ARGS=--port 8765 --open"
  echo Console URL: http://127.0.0.1:8765
) else (
  set "ARGS=%*"
)
echo Interpreter : %VPY%
echo Press Ctrl+C to stop.
echo.
"%VPY%" webui.py %ARGS%
set "RC=%ERRORLEVEL%"
echo.
echo Server stopped (exit code %RC%).
pause
exit /b %RC%
