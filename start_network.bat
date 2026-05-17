@echo off
chcp 65001 > nul
title Sonar — Network Mode

set ROOT=%~dp0

:: Find local IPv4 (first WiFi/Ethernet address)
set LOCAL_IP=
for /f "tokens=2 delims=:" %%A in ('ipconfig ^| findstr /C:"IPv4"') do (
    if not defined LOCAL_IP set LOCAL_IP=%%A
)
set LOCAL_IP=%LOCAL_IP: =%

echo.
echo  ==========================================
echo   SONAR — NETWORK MODE
echo  ==========================================
echo.
echo   PC (this machine):
echo   http://localhost:8765
echo.
echo   Phone / tablet (same WiFi):
echo   http://%LOCAL_IP%:8765
echo.
echo   Press Ctrl+C to stop the server.
echo  ==========================================
echo.

cd /d "%ROOT%track_a_desktop"
python "%ROOT%track_a_desktop\ui_local\app.py" --host 0.0.0.0 --port 8765

echo.
echo  [!] Server stopped or failed to start.
echo      Check errors above.
echo.
cmd /k
