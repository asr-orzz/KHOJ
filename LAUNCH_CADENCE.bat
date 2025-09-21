@echo off
echo ===============================================================================
echo 🚀 CADENCE AI System - Complete Setup and Launch
echo ===============================================================================
echo.
echo This will set up and run the complete CADENCE system:
echo.
echo ✅ Real Amazon QAC Dataset Processing (100K+ queries)
echo ✅ Real Amazon Products Dataset Processing (50K+ products)  
echo ✅ Advanced AI Clustering for Categories
echo ✅ Query Language Model Training
echo ✅ Catalog Language Model Training
echo ✅ Backend API Server (FastAPI)
echo ✅ React Frontend Integration
echo ✅ Complete Working Demo
echo.
echo 💻 Optimized for: RTX 3050 4GB + 16GB RAM + Ryzen 7 6800H
echo ⏱️  Setup time: 30-60 minutes
echo 💾 Storage: ~2-3 GB
echo.
echo ===============================================================================

:menu
echo.
echo SELECT AN OPTION:
echo ===============================================================================
echo 1. 🚀 Complete Setup (Recommended) - Full system from scratch
echo 2. 🔄 Quick Launch - Start existing system (if already set up)
echo 3. 🧠 Training Only - Just run data processing and model training
echo 4. 🌐 Servers Only - Start backend and frontend (if models exist)
echo 5. 🔧 Check System Status
echo 6. 📊 View System Requirements
echo 7. ❌ Exit
echo.
set /p choice="Enter your choice (1-7): "

if "%choice%"=="1" goto complete_setup
if "%choice%"=="2" goto quick_launch
if "%choice%"=="3" goto training_only
if "%choice%"=="4" goto servers_only
if "%choice%"=="5" goto check_status
if "%choice%"=="6" goto show_requirements
if "%choice%"=="7" goto exit
echo Invalid choice, please try again.
goto menu

:complete_setup
echo.
echo 🚀 STARTING COMPLETE CADENCE SETUP
echo ===============================================================================
echo This will run the full pipeline from scratch:
echo 1. Install Python dependencies
echo 2. Download and process Amazon datasets
echo 3. Train CADENCE AI models
echo 4. Setup React frontend
echo 5. Start backend API server
echo 6. Start frontend development server
echo 7. Launch complete working system
echo.
echo ⚠️  WARNING: This will take 30-60 minutes and requires internet connection!
echo.
set /p confirm="Are you sure you want to continue? (y/N): "
if /i not "%confirm%"=="y" goto menu

echo.
echo Starting complete setup...
python run_complete_cadence_system.py
goto end

:quick_launch
echo.
echo 🔄 QUICK LAUNCH
echo ===============================================================================
echo Checking if system is already set up...

if not exist "trained_models\real_cadence.pt" (
    echo ❌ Trained models not found!
    echo Please run Complete Setup first.
    pause
    goto menu
)

if not exist "processed_data\queries.parquet" (
    echo ❌ Processed data not found!
    echo Please run Complete Setup first.
    pause
    goto menu
)

echo ✅ System appears to be set up
echo Starting backend and frontend servers...

start "CADENCE Backend" python cadence_backend.py
timeout /t 5 /nobreak >nul

cd frontend
start "CADENCE Frontend" npm start
cd ..

echo.
echo ✅ CADENCE system launched!
echo.
echo 🌐 Open these URLs in your browser:
echo    Frontend: http://localhost:3000
echo    Backend: http://localhost:8000
echo    API Docs: http://localhost:8000/docs
echo.
echo Press any key to return to menu (servers will keep running)...
pause >nul
goto menu

:training_only
echo.
echo 🧠 TRAINING ONLY
echo ===============================================================================
echo This will only run data processing and model training.
echo Frontend integration will be skipped.
echo.
echo Step 1: Processing Amazon datasets...
python real_cadence_training.py

if errorlevel 1 (
    echo ❌ Data processing failed!
    pause
    goto menu
)

echo.
echo Step 2: Training CADENCE models...
python real_model_training.py

if errorlevel 1 (
    echo ❌ Model training failed!
    pause
    goto menu
)

echo.
echo ✅ Training completed successfully!
echo Models saved to: trained_models/
echo Data saved to: processed_data/
echo.
pause
goto menu

:servers_only
echo.
echo 🌐 SERVERS ONLY
echo ===============================================================================
echo Starting backend and frontend servers...

if not exist "trained_models\real_cadence.pt" (
    echo ❌ No trained models found!
    echo Please run training first.
    pause
    goto menu
)

echo Starting backend server...
start "CADENCE Backend" python cadence_backend.py
timeout /t 5 /nobreak >nul

echo Starting frontend server...
cd frontend
start "CADENCE Frontend" npm start
cd ..

echo.
echo ✅ Servers started!
echo    Backend: http://localhost:8000
echo    Frontend: http://localhost:3000
echo.
pause
goto menu

:check_status
echo.
echo 🔧 SYSTEM STATUS CHECK
echo ===============================================================================

echo Checking Python...
python --version
if errorlevel 1 (
    echo ❌ Python not found or not in PATH
) else (
    echo ✅ Python is available
)

echo.
echo Checking Node.js...
node --version
if errorlevel 1 (
    echo ❌ Node.js not found or not in PATH
) else (
    echo ✅ Node.js is available
)

echo.
echo Checking npm...
npm --version
if errorlevel 1 (
    echo ❌ npm not found or not in PATH
) else (
    echo ✅ npm is available
)

echo.
echo Checking required files...
if exist "real_cadence_training.py" (
    echo ✅ Data processing script found
) else (
    echo ❌ real_cadence_training.py missing
)

if exist "real_model_training.py" (
    echo ✅ Model training script found
) else (
    echo ❌ real_model_training.py missing
)

if exist "cadence_backend.py" (
    echo ✅ Backend script found
) else (
    echo ❌ cadence_backend.py missing
)

if exist "frontend\" (
    echo ✅ Frontend directory found
) else (
    echo ❌ frontend/ directory missing
)

echo.
echo Checking processed data...
if exist "processed_data\queries.parquet" (
    echo ✅ Processed queries found
) else (
    echo ❌ Processed queries not found
)

if exist "processed_data\products.parquet" (
    echo ✅ Processed products found
) else (
    echo ❌ Processed products not found
)

echo.
echo Checking trained models...
if exist "trained_models\real_cadence.pt" (
    echo ✅ Trained model found
) else (
    echo ❌ Trained model not found
)

if exist "trained_models\real_cadence_vocab.pkl" (
    echo ✅ Vocabulary found
) else (
    echo ❌ Vocabulary not found
)

echo.
echo Status check completed.
pause
goto menu

:show_requirements
echo.
echo 📊 SYSTEM REQUIREMENTS
echo ===============================================================================
echo.
echo MINIMUM REQUIREMENTS:
echo • Windows 10/11
echo • Python 3.8+ (with pip)
echo • Node.js 16+ (with npm)
echo • 8GB+ RAM (16GB recommended)
echo • 2GB+ GPU VRAM (4GB recommended) 
echo • 5GB+ free disk space
echo • Internet connection (for datasets)
echo.
echo RECOMMENDED SETUP (This System):
echo • RTX 3050 4GB GPU
echo • 16GB RAM
echo • Ryzen 7 6800H CPU
echo • Windows 11
echo • Fast SSD storage
echo.
echo PERFORMANCE EXPECTATIONS:
echo • Data processing: 10-20 minutes
echo • Model training: 15-30 minutes
echo • Autocomplete response: <100ms
echo • Search results: <200ms
echo.
echo WHAT YOU GET:
echo • Real Amazon QAC dataset (100K+ queries)
echo • Real Amazon Products dataset (50K+ products)
echo • Advanced AI clustering for categories
echo • Production-ready Query Language Model
echo • Production-ready Catalog Language Model  
echo • FastAPI backend with auto-docs
echo • Modern React frontend
echo • Complete working demo system
echo.
pause
goto menu

:end
echo.
echo 🎉 CADENCE system setup completed!
echo Check the output above for access URLs.
echo.
pause

:exit
echo.
echo 👋 Thank you for using CADENCE AI!
echo.
echo If you encountered any issues:
echo 1. Check system requirements
echo 2. Ensure stable internet connection
echo 3. Run as Administrator if needed
echo 4. Check Python and Node.js installations
echo.
pause