@echo off
echo =======================================================
echo 🚀 CADENCE Super-Optimized Training Launcher
echo =======================================================
echo.
echo This will start training in the background optimized for:
echo - RTX 3050 4GB GPU
echo - 16GB RAM  
echo - Ryzen 7 6800H CPU
echo - Windows 10/11
echo.
echo Features:
echo ✅ Memory-optimized model architecture
echo ✅ Mixed precision training (FP16)
echo ✅ Aggressive GPU memory management
echo ✅ Checkpointing with resume capability
echo ✅ Background training
echo ✅ Real-time memory monitoring
echo.

:menu
echo =======================================================
echo SELECT TRAINING MODE:
echo =======================================================
echo 1. Quick Test (5000 samples, 3 epochs) - ~15 minutes
echo 2. Standard Training (10000 samples, 5 epochs) - ~30 minutes
echo 3. Extended Training (20000 samples, 8 epochs) - ~60 minutes
echo 4. Custom Training (specify parameters)
echo 5. Resume Previous Training
echo 6. Check Training Status
echo 7. Stop Training
echo 8. Exit
echo.
set /p choice="Enter your choice (1-8): "

if "%choice%"=="1" goto quick
if "%choice%"=="2" goto standard
if "%choice%"=="3" goto extended
if "%choice%"=="4" goto custom
if "%choice%"=="5" goto resume
if "%choice%"=="6" goto status
if "%choice%"=="7" goto stop
if "%choice%"=="8" goto exit
echo Invalid choice, please try again.
goto menu

:quick
echo Starting Quick Test Training...
python run_optimized_training.py start --epochs 3 --dataset-size 5000
goto post_start

:standard
echo Starting Standard Training...
python run_optimized_training.py start --epochs 5 --dataset-size 10000
goto post_start

:extended
echo Starting Extended Training...
python run_optimized_training.py start --epochs 8 --dataset-size 20000
goto post_start

:custom
echo.
set /p epochs="Enter number of epochs (default 5): "
if "%epochs%"=="" set epochs=5
set /p dataset_size="Enter dataset size (default 10000): "
if "%dataset_size%"=="" set dataset_size=10000
echo Starting Custom Training...
python run_optimized_training.py start --epochs %epochs% --dataset-size %dataset_size%
goto post_start

:resume
echo Resuming Previous Training...
python run_optimized_training.py start
goto post_start

:status
echo.
python run_optimized_training.py status
echo.
pause
goto menu

:stop
echo.
python run_optimized_training.py stop
echo.
pause
goto menu

:post_start
echo.
echo =======================================================
echo 🎯 TRAINING STARTED IN BACKGROUND!
echo =======================================================
echo.
echo What you can do now:
echo ✅ Close this window - training continues in background
echo ✅ Use your computer normally - training won't interfere
echo ✅ Check progress anytime with: start_training.bat
echo.
echo Monitoring commands:
echo • python run_optimized_training.py status   (check once)
echo • python run_optimized_training.py monitor  (real-time)
echo • python run_optimized_training.py stop     (stop training)
echo.
echo Model will be saved to: models/optimized_cadence.pt
echo Checkpoints saved to: checkpoints/
echo.
set /p continue="Press 1 to monitor now, or any key to exit: "
if "%continue%"=="1" (
    python run_optimized_training.py monitor
)
goto exit

:exit
echo.
echo 👋 Thanks for using CADENCE Training!
echo Don't forget to check your training progress later.
pause