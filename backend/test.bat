@echo off
REM test.bat - Test the complete system on Windows

setlocal enabledelayedexpansion

echo.
echo 🧪 Testing Customer Churn Prediction System...
echo.

REM Color simulation (Windows doesn't support ANSI by default)
REM Using symbols instead

REM Check Python environment
echo ========== CHECKING PYTHON ENVIRONMENT ==========
echo.

cd backend

if not exist "venv" (
    echo ❌ Virtual environment not found
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo Python version:
python --version

echo Pip version:
pip --version

echo.
echo Checking required packages...
echo.

setlocal enabledelayedexpansion
for %%p in (pandas scikit-learn xgboost fastapi uvicorn streamlit numpy) do (
    python -c "import %%p" 2>nul
    if !errorlevel! equ 0 (
        echo ✓ %%p
    ) else (
        echo ✗ %%p
    )
)

echo.
echo ========== CHECKING MODEL FILES ==========
echo.

if exist "models\best_model.pkl" (
    echo ✓ Best model found
) else (
    echo ✗ Best model not found
)

if exist "models\preprocessor.pkl" (
    echo ✓ Preprocessor found
) else (
    echo ✗ Preprocessor not found
)

if exist "models\metrics.pkl" (
    echo ✓ Metrics found
) else (
    echo ✗ Metrics not found
)

echo.
echo ========== TESTING API ==========
echo.

echo Make sure API server is running in another terminal:
echo   cd backend
echo   venv\Scripts\activate
echo   uvicorn api.main:app --reload
echo.

echo Testing API endpoints...
echo.

REM Test health check
echo 1. Testing health check...
powershell -Command "(Invoke-WebRequest -Uri 'http://127.0.0.1:8000/health' -ErrorAction SilentlyContinue).Content" > api_response.txt
findstr /I "healthy" api_response.txt >nul
if !errorlevel! equ 0 (
    echo ✓ Health check passed
) else (
    echo ✗ Health check failed or API not running
)

echo.
echo 2. Testing prediction endpoint...

REM Create test data JSON
(
echo {
echo   "gender": "Female",
echo   "SeniorCitizen": 0,
echo   "Partner": "Yes",
echo   "Dependents": "No",
echo   "tenure": 12,
echo   "PhoneService": "Yes",
echo   "MultipleLines": "No",
echo   "InternetService": "DSL",
echo   "OnlineSecurity": "Yes",
echo   "OnlineBackup": "No",
echo   "DeviceProtection": "No",
echo   "TechSupport": "Yes",
echo   "StreamingTV": "No",
echo   "StreamingMovies": "No",
echo   "Contract": "Two year",
echo   "PaperlessBilling": "Yes",
echo   "PaymentMethod": "Credit card",
echo   "MonthlyCharges": 85.5,
echo   "TotalCharges": 1026.0
echo }
) > test_data.json

powershell -Command "(Invoke-WebRequest -Uri 'http://127.0.0.1:8000/predict' -Method Post -ContentType 'application/json' -InFile 'test_data.json' -ErrorAction SilentlyContinue).Content" > api_response.txt

findstr /I "prediction" api_response.txt >nul
if !errorlevel! equ 0 (
    echo ✓ Prediction endpoint works
) else (
    echo ✗ Prediction endpoint failed
)

echo.
echo 3. Testing model info endpoint...
powershell -Command "(Invoke-WebRequest -Uri 'http://127.0.0.1:8000/model/info' -ErrorAction SilentlyContinue).Content" > api_response.txt
findstr /I "model" api_response.txt >nul
if !errorlevel! equ 0 (
    echo ✓ Model info endpoint works
) else (
    echo ✗ Model info endpoint failed
)

echo.
echo ========== TESTING FRONTEND ==========
echo.

echo Make sure Next.js is running in another terminal:
echo   cd frontend
echo   npm run dev
echo.

echo Checking if frontend is accessible...
powershell -Command "(Invoke-WebRequest -Uri 'http://localhost:3000' -ErrorAction SilentlyContinue).Content" > frontend_response.txt
findstr /I "html" frontend_response.txt >nul
if !errorlevel! equ 0 (
    echo ✓ Frontend is accessible
) else (
    echo ⚠ Frontend not running or not responding
)

echo.
echo ========== TEST COMPLETE ==========
echo.

REM Cleanup
del api_response.txt 2>nul
del frontend_response.txt 2>nul
del test_data.json 2>nul

cd ..
pause
