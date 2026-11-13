@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 goto use_python
py -3.12 run.py %*
set "VTC_EXIT=%ERRORLEVEL%"
goto done
:use_python
python run.py %*
set "VTC_EXIT=%ERRORLEVEL%"
:done
if not "%VTC_NO_PAUSE%"=="1" pause
exit /b %VTC_EXIT%
