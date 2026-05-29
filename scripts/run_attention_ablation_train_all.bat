@echo off
setlocal

cd /d G:\XR_Xray_Detection

conda run --no-capture-output -n pg_moe python scripts\run_attention_ablation_train_all.py
if errorlevel 1 goto fail

exit /b 0

:fail
echo.
echo ============================================================
echo Attention ablation training stopped because the Python runner failed.
echo Check the console output above and the current run directory.
echo ============================================================
exit /b 1
