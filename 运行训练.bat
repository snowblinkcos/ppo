@echo off
chcp 65001 >nul
echo ========================================
echo   PPO 能量聚焦训练
echo   参数请在 config.py 中修改
echo ========================================
echo.

cd /d "%~dp0"
"C:\Users\李浩镔\AppData\Local\Programs\Python\Python312\python.exe" train_energy_focusing_ppo.py

echo.
echo ========================================
echo   训练完成！结果保存在 results 文件夹
echo ========================================
pause
