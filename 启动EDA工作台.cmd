@echo off
setlocal
set "edaDirectory=%~dp0"
if exist "%edaDirectory%eda_gui.exe" goto launch
set "edaDirectory=%~dp0dist\v0.3.1\EDA_Toolkit-v0.3.1\"
if exist "%edaDirectory%eda_gui.exe" goto launch
echo EDA Toolkit executable was not found. Extract the complete portable package first. 1>&2
exit /b 1

:launch
start "" /D "%edaDirectory%" "%edaDirectory%eda_gui.exe" %*
exit /b %errorlevel%
