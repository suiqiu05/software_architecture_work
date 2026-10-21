@echo off
rem ============================================================
rem  setup_env.bat - create .venv and install dependencies
rem  Portable: locates Python via `py -3` / `python` / `python3`
rem  and validates it really runs (Windows Store stubs exit silently).
rem ============================================================
setlocal
cd /d "%~dp0"

echo ============================================================
echo   AI-MAPE-K Controller - environment setup
echo ============================================================
echo.

rem ---- 1. find a real interpreter ----
set "BASE="
call :probe "py -3"
if not defined BASE call :probe "python"
if not defined BASE call :probe "python3"
if not defined BASE goto :no_python
echo [1/3] base interpreter : %BASE%

rem ---- 2. create project-local venv ----
set "VPY=%~dp0.venv\Scripts\python.exe"
if exist "%VPY%" (
  echo [2/3] .venv already exists - reusing
) else (
  echo [2/3] creating .venv ...
  "%BASE%" -m venv "%~dp0.venv"
  if errorlevel 1 (
    echo [ERROR] venv creation failed. Is venv/ensurepip installed?
    pause
    exit /b 1
  )
)
if not exist "%VPY%" (
  echo [ERROR] .venv\Scripts\python.exe missing after venv creation.
  pause
  exit /b 1
)

rem ---- 3. install dependencies ----
echo [3/3] installing numpy + matplotlib ...
"%VPY%" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
  echo [WARN] default index failed - retrying via Aliyun mirror ...
  "%VPY%" -m pip install --disable-pip-version-check --proxy "" ^
      -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
  if errorlevel 1 (
    echo [ERROR] dependency installation failed. Check network / proxy.
    pause
    exit /b 1
  )
)

"%VPY%" -c "import numpy, matplotlib; print('OK  numpy', numpy.__version__, '| matplotlib', matplotlib.__version__)"
echo.
echo Setup complete. Next: run.bat   (or double-click it)
pause
exit /b 0

:probe
for /f "delims=" %%i in ('%~1 -c "import sys; print(sys.executable)" 2^>nul') do (
  if not defined BASE set "BASE=%%i"
)
exit /b 0

:no_python
echo [ERROR] No working Python interpreter found.
echo         - Install Python 3.9+ from https://www.python.org/downloads/
echo         - Make sure "Add python.exe to PATH" is checked.
echo         - On Windows, also disable the Microsoft Store alias for
echo           python.exe / python3.exe (it is a 0-byte stub that exits
echo           silently): Settings ^> Apps ^> Advanced app settings
echo           ^> App execution aliases.
pause
exit /b 1
