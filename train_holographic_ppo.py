"""
train_holographic_ppo.py
========================
复现论文 Fig. 5 —— 全息图像生成实验（PPO 算法）

  "Model-free optical processors using in situ reinforcement learning
   with proximal policy optimization"  Li et al., Light: Sci. & Appl. (2026)

实验设置 (Fig. 5a):
  - 相位型 SLM 调制平面波相位
  - ASM 传播到传感器平面，形成全息图像
  - 奖励信号：输出强度图与目标图像的 MSE（评估指标：PSNR）

目标图像 (Fig. 5b/c/d):
  - 合成光栅 (Grating)：正弦条纹图案
  - 自然图像 (Boat)：scipy.datasets.ascent 标准测试图

物理参数:
  λ = 520 nm,  z = 9.6 cm,  pixel_size = 16 μm,  SLM = 128×128

运行方法:
    python train_holographic_ppo.py
    python train_holographic_ppo.py --n_iter 2000 --M 256
"""

import os
import argparse
import math
import time

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from optical_sim import asm_propagate
from ppo import GaussianPolicy, PPOTrainer


# ---------------------------------------------------------------------------
# 物理常数（与能量聚焦实验相同）
# ---------------------------------------------------------------------------
WAVELENGTH = 520e-9    # 520 nm
Z          = 9.6e-2    # 9.6 cm 传播距离
PIXEL_SIZE = 16e-6     # 像素间距
LAYER_SIZE = 128      # 传感器平面尺寸


# ---------------------------------------------------------------------------
# 目标图像生成
# ---------------------------------------------------------------------------

def make_grating(H: int, W: int, freq: int = 8) -> torch.Tensor:
    """
    合成正弦竖条纹光栅，归一化到 [0, 1]。
    freq: 水平方向完整周期数。对应论文 Fig. 5c/d 中的 Grating 目标。
    """
    x = torch.linspace(0.0, 2.0 * math.pi * freq, W) ##linspace(0, 2π*freq, W)
    col = (torch.sin(x) + 1.0) / 2.0 # 归一化到 [0, 1]
    return col.unsqueeze(0).expand(H, -1).clone()   # (H, W)


def make_boat_target(H: int, W: int) -> torch.Tensor:  
    """
    加载自然图像目标 ("Boat")。
    使用 scipy.datasets.ascent 标准测试图（512×512 灰度楼梯图），
    缩放到 (H, W) 并归一化到 [0, 1]。
    """
    from scipy.datasets import ascent  #ascent() 返回一个 512×512 的 uint8 数组，值在 [0,255] 之间
    arr = ascent().astype(np.float32)      # 转为 float32
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)   # (1,1,512,512)
    t = F.interpolate(t, size=(H, W), mode='bilinear',
                      align_corners=False).squeeze()       #缩小到 (H, W)
    return (t / t.max()).clamp(0.0, 1.0) # 归一化到 [0, 1]，clamp小于0的值为0，大于1的值为1


# ---------------------------------------------------------------------------
# PSNR 指标
# ---------------------------------------------------------------------------

def compute_psnr(output: torch.Tensor, target: torch.Tensor) -> float:
    """
    计算 PSNR (dB)。
    output 先按自身最大值归一化到 [0,1]，再与 target 对比。
    """
    out = output.float() / (output.max().clamp(min=1e-12))
    mse = F.mse_loss(out.cpu(), target.cpu().float()).item()#计算均方误差 mse_loss = F.mse_loss(out, target)  # MSE 损失
    if mse < 1e-12:#loss极小的时候
        return 40.0 # 40 dB 以上视为近乎完美重建
    return 10.0 * math.log10(1.0 / mse)


# ---------------------------------------------------------------------------
# 全息成像任务
# ---------------------------------------------------------------------------

class HolographicTask:
    """
    Phase-only SLM → ASM 传播 → 传感器平面强度 vs 目标图像。

    训练损失: MSE(归一化输出强度, 目标图像)  —— 作为 PPO 的奖励信号
    评估指标: PSNR (dB)
    """

    def __init__(self, target: torch.Tensor, device: str):
        self.target = target.to(device)
        self.device = device
        self.plane_wave = torch.ones(
            LAYER_SIZE, LAYER_SIZE, dtype=torch.complex64, device=device
        )

    def _propagate_batch(self, phase_masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            phase_masks : float (M, H, W)
        Returns:
            intensities : float (M, H, W)，每个样本独立归一化到 [0,1]
        """
        phi = phase_masks.to(self.device)
        fields = self.plane_wave.unsqueeze(0) * torch.exp(
            1j * phi.float() #相位掩模转换为复数形式
        ).to(torch.complex64)
        out = asm_propagate(fields, Z, WAVELENGTH, PIXEL_SIZE) #ASM 传播，返回 (M, H, W) 复数场
        intensity = out.abs() ** 2
        mx = intensity.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-12)#每个样本的最大强度，防止除零
        return intensity / mx

    @torch.no_grad()
    def compute_losses(self, phase_masks: torch.Tensor) -> torch.Tensor:
        """MSE 损失，返回 (M,) CPU 张量，作为 PPO 奖励信号。"""
        intensities = self._propagate_batch(phase_masks)#(1) 传播得到 (M, H, W) 强度图
        tgt = self.target.unsqueeze(0).expand_as(intensities)#(2) 扩展目标图像到 (M, H, W)
        return (intensities - tgt).pow(2).mean(dim=(-2, -1)).cpu()#(3) 计算每个样本的 MSE 损失，返回 (M,) CPU 张量

    @torch.no_grad()
    def eval_psnr(self, phase_mask: torch.Tensor) -> float:
        """对当前均值相位（无噪声）评估 PSNR。"""
        intensity = self._propagate_batch(phase_mask.unsqueeze(0))[0]
        return compute_psnr(intensity, self.target)

    @torch.no_grad()
    def get_output(self, phase_mask: torch.Tensor) -> torch.Tensor:
        """返回单个相位掩模对应的归一化强度图 (H,W)，CPU 张量。"""
        return self._propagate_batch(phase_mask.unsqueeze(0))[0].cpu()


# ---------------------------------------------------------------------------
# 训练循环
# ---------------------------------------------------------------------------

def train(task, policy, trainer, n_iter, M, eval_every, snap_iters):
    """
    执行 PPO 训练循环。

    Returns:
        times_sec : list[float] —— 每次评估的 wall-clock 时间 (s)
        psnr_vals : list[float] —— 对应的 PSNR (dB)
        snapshots : list[(iter, psnr, output_img)] —— 指定迭代快照
    """
    times_sec  = []
    psnr_vals  = []
    snapshots  = []
    seen_snaps = set()
    t0 = time.time()

    for it in range(1, n_iter + 1):
        phi    = policy.sample(M)
        losses = task.compute_losses(phi)
        trainer.update(phi, losses)

        if it % eval_every == 0 or it in snap_iters:
            mean_phase = policy.get_mean()
            p = task.eval_psnr(mean_phase)
            t = time.time() - t0
            times_sec.append(t)
            psnr_vals.append(p)

            if it in snap_iters and it not in seen_snaps:
                snapshots.append((it, p, task.get_output(mean_phase).clone()))
                seen_snaps.add(it)

            if it % (eval_every * 10) == 0 or it == 1 or it == n_iter:
                print(f"  iter={it:4d}  t={t:.1f}s  PSNR={p:.2f} dB")

    return times_sec, psnr_vals, snapshots


# ---------------------------------------------------------------------------
# 绘图函数
# ---------------------------------------------------------------------------

def plot_psnr_curve(times, psnr_vals, target_name, save_path): 
    """Fig. 5b：单目标的 PPO PSNR 随训练时间变化曲线。"""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(times, psnr_vals, color='tab:orange', linewidth=2, label='PPO')
    ax.set_xlabel('Training Time (s)', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title(f'Reconstruction quality — {target_name}  (Fig. 5b)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved PSNR curve    → {save_path}")


def plot_combined_psnr(results, save_path):
    """Fig. 5b：两目标合并在一张图上。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {'Grating': '--', 'Boat': '-'}
    for tname, (t_vals, p_vals, _) in results.items():
        ax.plot(t_vals, p_vals, color='tab:orange',
                linestyle=styles[tname], linewidth=2, label=f'PPO {tname}')
    ax.set_xlabel('Training Time (s)', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title('Experimental reconstruction quality (Fig. 5b)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved combined PSNR → {save_path}")


def plot_snapshots(target, snapshots, target_name, save_path):
    """
    Fig. 5c：PPO 在 4 个迭代点的全息重建快照。

    布局:  Target | snap1 | snap2 | snap3 | snap4
    """
    n = len(snapshots)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n + 1, figsize=(2.4 * (n + 1), 3.2))

    tgt_np = target.cpu().float().numpy()
    tgt_np = tgt_np / (tgt_np.max() + 1e-12)

    def show(ax, img, title):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1,
                  interpolation='nearest', aspect='equal')
        ax.set_title(title, fontsize=8, pad=3)
        ax.axis('off')

    show(axes[0], tgt_np, f'Target\n({target_name})')
    for j, (it, p, out) in enumerate(snapshots):
        img = out.float().numpy()
        img = img / (img.max() + 1e-12)
        show(axes[j + 1], img, f'PPO  iter={it}\nPSNR={p:.1f} dB')

    fig.suptitle(f'Holographic image generation — {target_name}  (Fig. 5c)',
                 fontsize=11)
    plt.tight_layout() #调整子图间距
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved snapshots     → {save_path}")


def plot_cross_section(target, ppo_out, save_path):
    """
    Fig. 5d：光栅图案水平截面强度轮廓，对比 Target vs PPO。
    Michelson 对比度 C = (I_max - I_min) / (I_max + I_min)。
    """
    row = LAYER_SIZE // 2

    def norm1d(t): #提取指定行并归一化到 [0, 1]
        a = t.float().numpy()[row, :]
        return a / (a.max() + 1e-12)

    def contrast(a): #计算 Michelson 对比度
        mx, mn = a.max(), a.min()
        return float((mx - mn) / (mx + mn + 1e-12))

    t_arr   = norm1d(target)
    ppo_arr = norm1d(ppo_out)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    x = np.arange(LAYER_SIZE)

    for ax, arr, label, col in zip(
            axes,
            [t_arr, ppo_arr],
            [f'Target  (C={contrast(t_arr):.2f})',
             f'PPO     (C={contrast(ppo_arr):.2f})'],
            ['black', 'tab:orange']):
        ax.plot(x, arr, color=col, linewidth=1.5)
        ax.set_title(label, fontsize=10)
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlabel('Pixel index', fontsize=9)
        ax.grid(True, alpha=0.3)#添加网格线
    axes[0].set_ylabel('Normalized intensity', fontsize=9)#添加 y 轴标签

    fig.suptitle('Cross-sectional intensity profile — Grating  (Fig. 5d)',
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved cross-section → {save_path}")


def save_phase_pattern(phase, path, title):
    """保存学习到的相位掩模（对应论文 Fig. 5a 左侧的 Trainable phase pattern）。"""
    arr = phase.detach().cpu().numpy() % (2 * math.pi)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(arr, cmap='hsv', vmin=0, vmax=2 * math.pi,
                   interpolation='nearest')
    fig.colorbar(im, ax=ax, fraction=0.046).set_label('Phase (rad)')
    ax.set_title(title, fontsize=9)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='全息图像生成实验复现 (论文 Fig. 5)')
    p.add_argument('--n_iter',         type=int,   default=10000)
    p.add_argument('--M',              type=int,   default=256,
                   help='每轮采样数（论文物理：32；仿真推荐：256）')
    p.add_argument('--K',              type=int,   default=4,
                   help='PPO 内循环更新次数（论文：4）')
    p.add_argument('--sigma',          type=float, default=0.3)
    p.add_argument('--lr',             type=float, default=0.3)
    p.add_argument('--epsilon',        type=float, default=0.02,
                   help='PPO 截断参数 ε')
    p.add_argument('--pixel_grouping', type=int,   default=4,
                   help='像素分组大小 g；g=1 → DOF=256')
    p.add_argument('--eval_every',     type=int,   default=20)
    p.add_argument('--save_dir',       type=str,   default='./results')
    p.add_argument('--seed',           type=int,   default=42)
    p.add_argument('--device',         type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA 不可用，回退到 CPU")
        device = 'cpu'

    dof = (LAYER_SIZE // args.pixel_grouping) ** 2
    print(f"\n{'='*60}")
    print(f"  任务   : 全息图像生成  (论文 Fig. 5)")
    print(f"  设备   : {device}")
    print(f"  n_iter={args.n_iter}  M={args.M}  K={args.K}")
    print(f"  sigma={args.sigma}  lr={args.lr}  epsilon={args.epsilon}")
    print(f"  pixel_grouping={args.pixel_grouping}  (DOF={dof})")
    print(f"{'='*60}\n")

    snap_iters = {1,
                  max(1, args.n_iter // 4),
                  max(1, args.n_iter // 2),
                  args.n_iter}

    targets = {
        'Grating': make_grating(LAYER_SIZE, LAYER_SIZE, freq=8),
        'Boat':    make_boat_target(LAYER_SIZE, LAYER_SIZE),
    }

    all_results = {}   # target_name → (times, psnr_vals, snapshots)

    for target_name, target_img in targets.items():
        print(f"{'─'*50}")
        print(f"  目标: {target_name}")
        print(f"{'─'*50}")

        task    = HolographicTask(target_img, device=device)
        policy  = GaussianPolicy(LAYER_SIZE, LAYER_SIZE,
                                 sigma=args.sigma,
                                 pixel_grouping=args.pixel_grouping,
                                 device=device)
        trainer = PPOTrainer(policy, lr=args.lr,
                             epsilon=args.epsilon, K=args.K)

        t_vals, p_vals, snaps = train(
            task, policy, trainer,
            args.n_iter, args.M, args.eval_every, snap_iters)

        all_results[target_name] = (t_vals, p_vals, snaps)

        # ── PSNR 曲线 (Fig. 5b) ──────────────────────────────────────
        plot_psnr_curve(
            t_vals, p_vals, target_name,
            save_path=os.path.join(
                args.save_dir,
                f'holographic_{target_name.lower()}_psnr.png'))

        # ── 快照 (Fig. 5c) ────────────────────────────────────────────
        plot_snapshots(
            target_img, snaps, target_name,
            save_path=os.path.join(
                args.save_dir,
                f'holographic_{target_name.lower()}_snapshots.png'))

        # ── 最终相位掩模 ──────────────────────────────────────────────
        save_phase_pattern(
            policy.get_mean(),
            path=os.path.join(args.save_dir,
                              f'holographic_{target_name.lower()}_phase.png'),
            title=f'PPO final phase — {target_name}')

        print(f"  最终 PSNR: {p_vals[-1]:.2f} dB  (best: {max(p_vals):.2f} dB)\n")

    # ── 合并 PSNR 曲线 (Fig. 5b) ─────────────────────────────────────
    plot_combined_psnr(
        all_results,  # 传入所有结果
        save_path=os.path.join(args.save_dir, 'holographic_combined_psnr.png'))

    # ── 光栅截面轮廓 (Fig. 5d) ────────────────────────────────────────
    g_times, g_pvals, g_snaps = all_results['Grating']
    task_g  = HolographicTask(targets['Grating'], device=device)
    # 用最后一个快照的输出作为截面图
    if g_snaps:
        _, _, ppo_out = g_snaps[-1]
    else:
        # fallback：重新推理一次
        pol_g = GaussianPolicy(LAYER_SIZE, LAYER_SIZE,
                               sigma=args.sigma,
                               pixel_grouping=args.pixel_grouping,
                               device=device)
        ppo_out = task_g.get_output(pol_g.get_mean())

    plot_cross_section(
        targets['Grating'], ppo_out,
        save_path=os.path.join(args.save_dir,
                               'holographic_grating_cross_section.png'))

    # ── 汇总 ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  最终 PSNR 汇总")
    for tname, (_, p_vals, _) in all_results.items():
        if p_vals:
            print(f"  {tname:<10}  final={p_vals[-1]:.2f} dB  "
                  f"best={max(p_vals):.2f} dB")
    print(f"{'='*60}")
    print(f"\n结果已保存到: {os.path.abspath(args.save_dir)}\n")

    with open(os.path.join(args.save_dir, 'holographic_summary.txt'),
              'w', encoding='utf-8') as f:
        f.write(f"n_iter={args.n_iter}, M={args.M}, K={args.K}, "
                f"sigma={args.sigma}, lr={args.lr}, epsilon={args.epsilon}\n\n")
        for tname, (_, p_vals, _) in all_results.items():
            if p_vals:
                f.write(f"{tname}: final={p_vals[-1]:.2f} dB, "
                        f"best={max(p_vals):.2f} dB\n")


if __name__ == '__main__':
    main()
