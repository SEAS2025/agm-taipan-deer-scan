@echo off
title AGM Deer Scan
cd /d "%~dp0"

echo ============================================================
echo  AGM Taipan Deer Scanner
echo  Connect laptop to Taipan WiFi hotspot first.
echo ============================================================
echo.

python agm_deer_scanner.py
set EXITCODE=%ERRORLEVEL%
if %EXITCODE% NEQ 0 (
    echo.
    echo Scanner exited with error code %EXITCODE%.
    pause
)
