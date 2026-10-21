@echo off
rem ============================================================
rem  pack.bat - build a portable .zip of this project
rem  (excludes .venv / results / caches, so no machine-specific
rem   absolute paths end up in the archive)
rem ============================================================
setlocal
cd /d "%~dp0"

set "PYEXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYEXE%" (
  set "PYEXE="
  for /f "delims=" %%i in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do (
    if not defined PYEXE set "PYEXE=%%i"
  )
)
if not defined PYEXE (
  for /f "delims=" %%i in ('python -c "import sys; print(sys.executable)" 2^>nul') do (
    if not defined PYEXE set "PYEXE=%%i"
  )
)
if not defined PYEXE (
  echo [ERROR] No Python interpreter found to run pack.py
  pause
  exit /b 1
)

"%PYEXE%" pack.py %*
echo.
pause
