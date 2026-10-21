@echo off
rem ============================================================
rem  run.bat - run the experiment (portable, no absolute paths)
rem
rem  Interpreter resolution order:
rem    1) .\\.venv\\Scripts\\python.exe        (project-local venv, preferred)
rem    2) py -3 / python / python3             (system interpreter that
rem                                             already has numpy+matplotlib)
rem  If none is usable it tells you to run setup_env.bat first.
rem ============================================================
setlocal
cd /d "%~dp0"

set "VPY=%~dp0.venv\Scripts\python.exe"
if exist "%VPY%" goto :run

echo [info] .venv not found - probing system interpreters for numpy ...
set "VPY="
call :probe "py -3"
if not defined VPY call :probe "python"
if not defined VPY call :probe "python3"
if defined VPY goto :run

echo.
echo [ERROR] No usable Python environment found.
echo         Please run setup_env.bat first.
echo         (It creates .venv and installs numpy + matplotlib.)
pause
exit /b 1

:probe
rem %~1 expands to e.g.  py -3   (quotes stripped so args stay separate)
for /f "delims=" %%i in ('%~1 -c "import numpy,sys; print(sys.executable)" 2^>nul') do (
  if not defined VPY set "VPY=%%i"
)
exit /b 0

:run
echo Using interpreter: %VPY%
echo.
"%VPY%" main_exp.py %*
set "RC=%ERRORLEVEL%"
echo.
echo Exit code: %RC%
pause
exit /b %RC%
