@echo off
REM ============================================
REM Voyager Evolved - Installation Script for Windows
REM ============================================

echo ============================================
echo   Voyager Evolved Installation Script
echo ============================================
echo.

REM Check Python version
echo Checking Python version...
python --version 2>nul
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://www.python.org/
    pause
    exit /b 1
)

REM Check Node.js
echo Checking Node.js version...
node --version 2>nul
if errorlevel 1 (
    echo [WARNING] Node.js is not installed
    echo Please install Node.js from https://nodejs.org/
    pause
)

REM Create virtual environment
echo.
set /p CREATE_VENV="Create a virtual environment? (Y/n): "
if /i "%CREATE_VENV%"=="" set CREATE_VENV=Y
if /i "%CREATE_VENV%"=="Y" (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo [OK] Virtual environment created and activated
    echo.
    echo Remember to activate the venv with: venv\Scripts\activate.bat
)

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip
echo [OK] pip upgraded

REM Install Python package
echo.
echo Installing Voyager Evolved...
pip install -e .
echo [OK] Voyager Evolved installed

REM Install Mineflayer dependencies
echo.
echo Installing Mineflayer (Minecraft bot framework)...
if exist voyager\env\mineflayer (
    cd voyager\env\mineflayer
    npm install
    cd ..\..\..   
    echo [OK] Mineflayer installed
) else (
    echo [WARNING] Mineflayer directory not found, skipping npm install
)

REM Create config directory
if not exist configs mkdir configs

echo.
echo ============================================
echo [OK] Installation complete!
echo ============================================
echo.
echo Next steps:
echo   1. Set your OpenAI API key:
echo      set OPENAI_API_KEY=your-api-key-here
echo.
echo   2. Edit the config file:
echo      notepad configs\config.yaml
echo.
echo   3. Run Voyager Evolved:
echo      python run_voyager.py
echo.
echo For more info, see README.md
pause
