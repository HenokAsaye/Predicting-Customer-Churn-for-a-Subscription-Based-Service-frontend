@echo off
REM run.bat - Complete startup script for Windows development

echo.
echo 🚀 Starting Customer Churn Prediction System...
echo.

REM Check if running from correct directory
if not exist "backend" (
    echo ❌ Please run this script from the project root directory
    exit /b 1
)

echo.
echo ========== BACKEND SETUP ==========
echo 1. Checking Python environment...

REM Navigate to backend
cd backend

REM Create venv if not exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate venv
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Failed to activate virtual environment
    exit /b 1
)
echo ✓ Virtual environment activated

REM Install dependencies
echo 📦 Installing dependencies...
pip install -q -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install dependencies
    exit /b 1
)
echo ✓ Dependencies installed

REM Check if models exist
if not exist "models" (
    mkdir models
)
if not exist "models\best_model.pkl" (
    echo 📊 Training models...
    python train.py
    if errorlevel 1 (
        echo ❌ Model training failed
        exit /b 1
    )
    echo ✓ Models trained
) else (
    echo ✓ Models found
)

echo.
echo ========== FRONTEND SETUP ==========
cd ..\frontend

REM Check Node.js
where node >nul 2>nul
if errorlevel 1 (
    echo ❌ Node.js not found. Please install Node.js 16+
    exit /b 1
)

for /f "tokens=*" %%i in ('node -v') do set NODE_VERSION=%%i
echo ✓ Node.js found: %NODE_VERSION%

REM Install npm dependencies
if not exist "node_modules" (
    echo 📦 Installing npm dependencies...
    call npm install
    if errorlevel 1 (
        echo ❌ npm install failed
        exit /b 1
    )
    echo ✓ Dependencies installed
) else (
    echo ✓ node_modules found
)

echo.
echo ========== READY TO START ==========
echo.
echo Open 3 terminals and run:
echo.
echo Terminal 1 ^(API Server^):
echo   cd backend
echo   venv\Scripts\activate
echo   uvicorn api.main:app --reload
echo.
echo Terminal 2 ^(Next.js Frontend^):
echo   cd frontend
echo   npm run dev
echo.
echo Terminal 3 ^(Alternative - Streamlit^):
echo   cd backend
echo   venv\Scripts\activate
echo   streamlit run ..\app\streamlit_app.py
echo.
echo Then visit:
echo   Next.js:  http://localhost:3000
echo   Streamlit: http://localhost:8501
echo   API:      http://localhost:8000/docs
echo.
pause
