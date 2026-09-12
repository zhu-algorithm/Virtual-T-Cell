@echo off
cd /d "%~dp0"
python -m pip install -e ".[ui]"
if errorlevel 1 exit /b 1
python -m streamlit run app.py
