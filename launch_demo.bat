@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
python demo\demo.py
echo.
echo Demo exited. Press any key to close this window.
pause >nul
