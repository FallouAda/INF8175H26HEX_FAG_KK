@echo off
REM ============================================================
REM  HEX TOURNAMENT LAUNCHER
REM  Edit HOST and PORT below before the tournament, then
REM  just double-click this file to start playing.
REM ============================================================

set HOST=localhost
set PORT=16001

echo.
echo  ==============================
echo   HEX TOURNAMENT - AUTONOMOUS
echo   Server: %HOST%:%PORT%
echo  ==============================
echo.

cd /d "%~dp0"
python tournament_connect.py -a %HOST% -p %PORT%

pause
