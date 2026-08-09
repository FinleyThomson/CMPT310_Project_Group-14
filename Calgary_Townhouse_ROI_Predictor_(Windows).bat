@echo off
title ROI Predictor Server

::Navigate to folder with .bat
cd /d "%~dp0"

echo.
echo Starting ROI Predictor...
echo.
echo.

::check for venv, otherwise create it
IF NOT EXIST ".venv\" (
	echo First time setup, creating virtual environment...
	python -m venv .venv
	call ".venv\Scripts\activate.bat"
	pip install -r requirements.txt

	echo Setup complete!
	echo.
) ELSE (
	echo Virtual environment found, activating...
	call ".venv\Scripts\activate.bat"
)

::Move to website folder
cd Website

::Open the default web browser
echo Opening browser...
start http://127.0.0.1:8000/

::Start Flask server
echo Starting the server, do not close this window...
echo.
python interface.py

:: Move back up to the main folder
cd ..

:: If Python crashed, pause. If the Power Button was clicked, close 
IF %ERRORLEVEL% NEQ 0 (
    pause
)