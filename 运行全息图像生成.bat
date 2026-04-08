@echo off
chcp 65001 > nul
echo ============================================================
echo  全息图像生成实验  (论文 Fig. 5)  —  PPO
echo ============================================================
echo.

python train_holographic_ppo_yuzhe.py ^
  --n_iter 2000 ^
  --M 64 ^
  --K 4 ^
  --sigma 0.15 ^
  --sigma_final 0.05 ^
  --lr 0.2 ^
  --epsilon 0.02 ^
  --pixel_grouping 4 ^
  --eval_every 20

echo.
echo 结果已保存到 results_holographic\ 目录：
echo   holographic_combined_psnr.png          — Fig.5b 合并PSNR曲线
echo   holographic_grating_psnr.png           — Grating PSNR曲线
echo   holographic_boat_psnr.png              — Boat PSNR曲线
echo   holographic_grating_snapshots.png      — Fig.5c 光栅快照
echo   holographic_boat_snapshots.png         — Fig.5c Boat快照
echo   holographic_grating_cross_section.png  — Fig.5d 截面轮廓
echo   holographic_grating_phase.png          — 学到的光栅相位掩模
echo   holographic_boat_phase.png             — 学到的Boat相位掩模
echo   holographic_summary.txt               — 数值汇总

pause
