@echo off
title AGM Web Deer Scan
cd /d "%~dp0"

echo ============================================================
echo  AGM Taipan Web Scanner
echo  Opens http://127.0.0.1:8080 in your browser
echo  Connect laptop to Taipan WiFi hotspot first.
echo ============================================================
echo.

python agm_web_scanner.py --auto-train
set EXITCODE=%ERRORLEVEL%
if %EXITCODE% NEQ 0 (
    echo.
    echo Web scanner exited with error %EXITCODE%.
    pause
)
