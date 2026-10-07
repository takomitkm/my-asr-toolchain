@echo off
REM ---------------------------------------------------------------------------
REM setup.bat - Windows launcher for setup.py. ASCII only on purpose
REM (Windows .bat + codepage 936 mangles CJK).
REM
REM   setup.bat            runtime + builder deps (default)
REM   setup.bat nobuilder  skip requirements.txt (transcribe only, --graph int8)
REM   setup.bat noruntime  skip requirements-build.txt (builder env only)
REM   setup.bat --python "<path\to\python.exe>"     build the venv with that
REM   setup.bat --force                              skip the version-window check
REM
REM All real logic lives in setup.py, which is also what you run on Linux:
REM     python3 setup.py
REM Nothing here downloads models - that is fetch_assets.py (step 2).
REM Exit code 6 = environment problem, 0 = env is ready.
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

set "PY="
where py >nul 2>nul
if not errorlevel 1 set "PY=py -3"
if defined PY goto havepy
where python >nul 2>nul
if not errorlevel 1 set "PY=python"
:havepy
if not defined PY (
  echo [FAIL] no python found on PATH. Install Python 3.12 or 3.13, then rerun.
  echo        Or call it directly:  py -3.12 setup.py
  echo        Exit code 6 = environment problem.
  pause
  exit /b 6
)

%PY% setup.py %*
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
  echo setup.py finished, environment is ready.
) else (
  echo setup.py exited %RC% - read the lines above, 6 means the environment is wrong.
)
if /i not "%~1"=="--no-pause" pause
exit /b %RC%
