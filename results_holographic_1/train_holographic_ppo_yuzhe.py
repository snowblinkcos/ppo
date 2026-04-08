"""
train_holographic_ppo_yuzhe.py
==============================
复现论文 Fig. 5 —— 全息图像生成实验（PPO）

论文中与该任务直接相关的关键设置：
  - trainable phase pattern: 256×256, pixel size = 8 μm
  - target image: 128×128
  - propagation distance z = 9.6 cm, wavelength λ = 520 nm
  - measured image 先做能量归一化，再与目标图像计算 MSE
  - holographic image generation task 使用 M = 64

这个脚本和论文保持一致的部分：
  1. PPO 只优化 phase-only SLM。
  2. 传播使用 ASM。
  3. 损失和 PSNR 都基于“按目标总能量归一化后的强度图”。

运行示例:
    python train_holographic_ppo_yuzhe.py
    python train_holographic_ppo_yuzhe.py --n_iter 2000 --M 64 --pixel_grouping 4
"""

import argparse
import math
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from optical_sim import asm_propagate
from ppo import GaussianPolicy, PPOTrainer


# ---------------------------------------------------------------------------
# Paper-aligned physical constants (Fig. 5 / Methods)
# ---------------------------------------------------------------------------

WAVELENGTH = 520e-9
Z = 9.6e-2
PHASE_PIXEL_SIZE = 8e-6
PHASE_SIZE = 256
TARGET_SIZE = 128
TARGET_PIXEL_SIZE = 5.86e-6
TWO_PI = 2.0 * math.pi
CAMERA_CROP_SIZE = int(round(TARGET_SIZE * TARGET_PIXEL_SIZE / PHASE_PIXEL_SIZE))


# ---------------------------------------------------------------------------
# Target generation
# ---------------------------------------------------------------------------

def make_grating(H: int, W: int, freq: int = 8) -> torch.Tensor:
    """Synthetic vertical sinusoidal grating in [0, 1]."""
    x = torch.linspace(0.0, 2.0 * math.pi * freq, W)
    col = 0.5 * (torch.sin(x) + 1.0)
    return col.unsqueeze(0).expand(H, -1).clone()


def make_boat_target(H: int, W: int) -> torch.Tensor:
    """
    Load the Boat-like natural image target.

    Preferred source is scipy.datasets.ascent() because it is a standard
    grayscale benchmark image. If scipy is unavailable, fall back to the local
    example.png so the script remains runnable in a minimal environment.
    """
    arr = None
    try:
        from scipy.datasets import ascent
        arr = ascent().astype(np.float32)
    except ModuleNotFoundError:
        fallback = Path(__file__).resolve().parent / 'example.png'
        if not fallback.exists():
            raise RuntimeError(
                "scipy 未安装，且本地也没有 example.png 可作为 Boat 目标回退。"
            )
        from PIL import Image
        arr = np.array(Image.open(fallback).convert('L'), dtype=np.float32)
        print("  [warning] scipy 不可用，Boat 目标回退为本地 example.png 灰度图。")

    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(H, W), mode='bilinear', align_corners=False)
    t = t.squeeze(0).squeeze(0)
    return (t / 255.0).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Metrics / normalization
# ---------------------------------------------------------------------------

def normalize_measurement_to_target_energy(
    measurement: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Normalize each measured image so its total energy matches the target image.

    measurement : (..., H, W), non-negative
    target      : (H, W)
    """
    target_energy = target.sum().clamp(min=1e-12)
    meas_energy = measurement.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
    return measurement * (target_energy / meas_energy)


def compute_psnr(output: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR after energy normalization, matching the loss definition."""
    out = normalize_measurement_to_target_energy(output.float(), target.float())
    mse = F.mse_loss(out.cpu(), target.cpu().float()).item()
    if mse < 1e-12:
        return 80.0
    return 10.0 * math.log10(1.0 / mse)


# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------

class HolographicTask:
    """
    256×256 phase-only SLM -> ASM -> resize/camera plane -> energy-normalized MSE.

    The camera target in the paper is 128×128, while the trainable phase mask is
    256×256. We therefore propagate on the SLM grid and then convert the detected
    intensity to the target grid before computing the loss.
    """

    def __init__(
        self,
        target: torch.Tensor,
        device: str,
        use_diffuser: bool = True,
        diffuser_seed: int = 42,
    ):
        self.target = target.to(device)
        self.device = device
        self.phase_size = PHASE_SIZE
        self.target_size = target.shape[-1]
        self.plane_wave = torch.ones(
            self.phase_size,
            self.phase_size,
            dtype=torch.complex64,
            device=device,
        )

        self.use_diffuser = use_diffuser
        if use_diffuser:
            g = torch.Generator(device=device)
            g.manual_seed(diffuser_seed)
            self.diffuser_phase = TWO_PI * torch.rand(
                self.phase_size,
                self.phase_size,
                generator=g,
                device=device,
            )
        else:
            self.diffuser_phase = None

    def _camera_resize(self, intensity: torch.Tensor) -> torch.Tensor:
        """
        Map detector-plane intensity to the 128×128 camera target grid.

        The paper reports a 256×256 phase mask with 8 μm pitch, but the target
        image is sampled on a 128×128 camera grid with 5.86 μm pitch. That
        means the effective camera FOV is the central ~94×94 pixels on the
        propagation grid, not the whole 256×256 field compressed uniformly.
        """
        crop = min(CAMERA_CROP_SIZE, intensity.shape[-2], intensity.shape[-1])
        if crop > 0 and crop < intensity.shape[-1]:
            y0 = (intensity.shape[-2] - crop) // 2
            x0 = (intensity.shape[-1] - crop) // 2
            intensity = intensity[..., y0:y0 + crop, x0:x0 + crop]

        if intensity.shape[-1] == self.target_size and intensity.shape[-2] == self.target_size:
            return intensity
        resized = F.interpolate(
            intensity.unsqueeze(1),
            size=(self.target_size, self.target_size),
            mode='area',
        )
        return resized.squeeze(1)

    def _propagate_batch(self, phase_masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            phase_masks : (M, 256, 256)
        Returns:
            normalized_images : (M, 128, 128)
        """
        phi = torch.remainder(phase_masks.to(self.device), TWO_PI)
        fields = self.plane_wave.unsqueeze(0) * torch.exp(1j * phi).to(torch.complex64)

        if self.diffuser_phase is not None:
            diffuser = torch.exp(1j * self.diffuser_phase).to(torch.complex64)
            fields = fields * diffuser.unsqueeze(0)

        out = asm_propagate(fields, Z, WAVELENGTH, PHASE_PIXEL_SIZE)
        intensity = out.abs() ** 2
        intensity = self._camera_resize(intensity)
        return normalize_measurement_to_target_energy(intensity, self.target)

    @torch.no_grad()
    def compute_losses(self, phase_masks: torch.Tensor) -> torch.Tensor:
        images = self._propagate_batch(phase_masks)
        tgt = self.target.unsqueeze(0).expand_as(images)
        return (images - tgt).pow(2).mean(dim=(-2, -1)).cpu()

    @torch.no_grad()
    def eval_psnr(self, phase_mask: torch.Tensor) -> float:
        image = self._propagate_batch(phase_mask.unsqueeze(0))[0]
        return compute_psnr(image, self.target)

    @torch.no_grad()
    def get_output(self, phase_mask: torch.Tensor) -> torch.Tensor:
        return self._propagate_batch(phase_mask.unsqueeze(0))[0].cpu()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(task, policy, trainer, n_iter, M, eval_every, snap_iters, sigma_final=None):
    times_sec = []
    psnr_vals = []
    snapshots = []
    seen_snaps = set()
    t0 = time.time()

    sigma_init = policy.sigma
    do_anneal = sigma_final is not None and sigma_final < sigma_init

    for it in range(1, n_iter + 1):
        if do_anneal:
            alpha = (it - 1) / max(n_iter - 1, 1)
            policy.sigma = sigma_init * (1.0 - alpha) + sigma_final * alpha

        phi = policy.sample(M)
        losses = task.compute_losses(phi)
        trainer.update(phi, losses)

        if it % eval_every == 0 or it in snap_iters:
            mean_phase = policy.get_mean()
            psnr = task.eval_psnr(mean_phase)
            elapsed = time.time() - t0
            times_sec.append(elapsed)
            psnr_vals.append(psnr)

            if it in snap_iters and it not in seen_snaps:
                snapshots.append((it, psnr, task.get_output(mean_phase).clone()))
                seen_snaps.add(it)

            if it == 1 or it == n_iter or it % max(eval_every * 10, 1) == 0:
                sigma_msg = f"  sigma={policy.sigma:.3f}" if do_anneal else ""
                print(f"  iter={it:4d}  t={elapsed:.1f}s  PSNR={psnr:.2f} dB{sigma_msg}")

    policy.sigma = sigma_init
    return times_sec, psnr_vals, snapshots


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_psnr_curve(times, psnr_vals, target_name, save_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(times, psnr_vals, color='tab:orange', linewidth=2, label='PPO')
    ax.set_xlabel('Training Time (s)', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title(f'Reconstruction quality - {target_name} (Fig. 5b)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved PSNR curve    -> {save_path}")


def plot_combined_psnr(results, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {'Grating': '--', 'Boat': '-'}
    for tname, (t_vals, p_vals, _) in results.items():
        ax.plot(
            t_vals,
            p_vals,
            color='tab:orange',
            linestyle=styles.get(tname, '-'),
            linewidth=2,
            label=f'PPO {tname}',
        )
    ax.set_xlabel('Training Time (s)', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title('Experimental reconstruction quality (Fig. 5b)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved combined PSNR -> {save_path}")


def plot_snapshots(target, snapshots, target_name, save_path):
    n = len(snapshots)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n + 1, figsize=(2.5 * (n + 1), 3.2))
    tgt_np = target.cpu().float().numpy()

    def show(ax, img, title):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='nearest', aspect='equal')
        ax.set_title(title, fontsize=8, pad=3)
        ax.axis('off')

    show(axes[0], tgt_np, f'Target\n({target_name})')
    for j, (it, p, out) in enumerate(snapshots):
        img = out.float().numpy()
        img = img / max(float(img.max()), 1e-12)
        show(axes[j + 1], img, f'PPO iter={it}\nPSNR={p:.1f} dB')

    fig.suptitle(f'Holographic image generation - {target_name} (Fig. 5c)', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved snapshots     -> {save_path}")


def plot_cross_section(target, ppo_out, save_path):
    row = target.shape[0] // 2

    def norm1d(t):
        a = t.float().numpy()[row, :]
        return a / (a.max() + 1e-12)

    def contrast(a):
        mx, mn = a.max(), a.min()
        return float((mx - mn) / (mx + mn + 1e-12))

    t_arr = norm1d(target)
    ppo_arr = norm1d(ppo_out)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    x = np.arange(target.shape[1])

    labels = [
        f'Target (C={contrast(t_arr):.2f})',
        f'PPO (C={contrast(ppo_arr):.2f})',
    ]
    colors = ['black', 'tab:orange']
    arrays = [t_arr, ppo_arr]

    for ax, arr, label, col in zip(axes, arrays, labels, colors):
        ax.plot(x, arr, color=col, linewidth=1.5)
        ax.set_title(label, fontsize=10)
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlabel('Pixel index', fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('Normalized intensity', fontsize=9)

    fig.suptitle('Cross-sectional intensity profile - Grating (Fig. 5d)', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved cross-section -> {save_path}")


def save_phase_pattern(phase, path, title):
    arr = phase.detach().cpu().numpy() % TWO_PI
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(arr, cmap='hsv', vmin=0, vmax=TWO_PI, interpolation='nearest')
    fig.colorbar(im, ax=ax, fraction=0.046).set_label('Phase (rad)')
    ax.set_title(title, fontsize=9)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='全息图像生成实验复现 (论文 Fig. 5)')
    p.add_argument('--n_iter', type=int, default=2000)
    p.add_argument('--M', type=int, default=64,
                   help='每轮采样数；论文 Fig.5 holography 使用 M=64')
    p.add_argument('--K', type=int, default=4,
                   help='PPO 内循环更新次数')
    p.add_argument('--sigma', type=float, default=0.15)
    p.add_argument('--lr', type=float, default=0.2)
    p.add_argument('--epsilon', type=float, default=0.02,
                   help='PPO 截断参数 epsilon')
    p.add_argument('--sigma_final', type=float, default=0.05,
                   help='sigma 退火终点；默认从 0.15 退火到 0.05')
    p.add_argument('--pixel_grouping', type=int, default=4,
                   help='相位参数分组大小；256x256 时建议至少 4，避免 PPO SNR 过低')
    p.add_argument('--eval_every', type=int, default=20)
    p.add_argument('--save_dir', type=str, default='./results_holographic')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--targets', type=str, default='grating,boat',
                   help='逗号分隔：grating, boat')
    p.add_argument('--disable_diffuser', action='store_true',
                   help='关闭固定随机相位 diffuser')
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
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

    dof = (PHASE_SIZE // args.pixel_grouping) ** 2
    print(f"\n{'=' * 60}")
    print("  任务   : 全息图像生成 (论文 Fig. 5)")
    print(f"  设备   : {device}")
    print(f"  phase  : {PHASE_SIZE}x{PHASE_SIZE} @ {PHASE_PIXEL_SIZE * 1e6:.2f} um")
    print(f"  target : {TARGET_SIZE}x{TARGET_SIZE} @ {TARGET_PIXEL_SIZE * 1e6:.2f} um")
    print(f"  camera FOV on ASM grid : center {CAMERA_CROP_SIZE}x{CAMERA_CROP_SIZE}")
    print(f"  n_iter={args.n_iter}  M={args.M}  K={args.K}")
    print(f"  sigma={args.sigma}  lr={args.lr}  epsilon={args.epsilon}")
    print(f"  pixel_grouping={args.pixel_grouping}  (DOF={dof})")
    print(f"  diffuser={'off' if args.disable_diffuser else 'on'}")
    print(f"{'=' * 60}\n")

    snap_iters = {
        1,
        max(1, args.n_iter // 4),
        max(1, args.n_iter // 2),
        args.n_iter,
    }

    requested = [s.strip().lower() for s in args.targets.split(',') if s.strip()]
    targets = {}
    if 'grating' in requested:
        targets['Grating'] = make_grating(TARGET_SIZE, TARGET_SIZE, freq=8)
    if 'boat' in requested:
        targets['Boat'] = make_boat_target(TARGET_SIZE, TARGET_SIZE)
    if not targets:
        raise ValueError("--targets 至少需要包含 grating 或 boat")

    all_results = {}
    final_outputs = {}
    final_phases = {}

    for target_name, target_img in targets.items():
        print(f"{'-' * 50}")
        print(f"  目标: {target_name}")
        print(f"{'-' * 50}")

        task = HolographicTask(
            target_img,
            device=device,
            use_diffuser=not args.disable_diffuser,
            diffuser_seed=args.seed,
        )
        policy = GaussianPolicy(
            PHASE_SIZE,
            PHASE_SIZE,
            sigma=args.sigma,
            pixel_grouping=args.pixel_grouping,
            device=device,
        )
        trainer = PPOTrainer(policy, lr=args.lr, epsilon=args.epsilon, K=args.K)

        t_vals, p_vals, snaps = train(
            task,
            policy,
            trainer,
            args.n_iter,
            args.M,
            args.eval_every,
            snap_iters,
            sigma_final=args.sigma_final,
        )

        mean_phase = policy.get_mean()
        final_out = task.get_output(mean_phase)

        all_results[target_name] = (t_vals, p_vals, snaps)
        final_outputs[target_name] = final_out
        final_phases[target_name] = mean_phase.detach().cpu().clone()

        plot_psnr_curve(
            t_vals,
            p_vals,
            target_name,
            save_path=os.path.join(args.save_dir, f'holographic_{target_name.lower()}_psnr.png'),
        )
        plot_snapshots(
            target_img,
            snaps,
            target_name,
            save_path=os.path.join(args.save_dir, f'holographic_{target_name.lower()}_snapshots.png'),
        )
        save_phase_pattern(
            mean_phase,
            path=os.path.join(args.save_dir, f'holographic_{target_name.lower()}_phase.png'),
            title=f'PPO final phase - {target_name}',
        )

        if p_vals:
            print(f"  最终 PSNR: {p_vals[-1]:.2f} dB  (best: {max(p_vals):.2f} dB)\n")

    plot_combined_psnr(
        all_results,
        save_path=os.path.join(args.save_dir, 'holographic_combined_psnr.png'),
    )

    if 'Grating' in targets:
        plot_cross_section(
            targets['Grating'],
            final_outputs['Grating'],
            save_path=os.path.join(args.save_dir, 'holographic_grating_cross_section.png'),
        )

    print(f"\n{'=' * 60}")
    print("  最终 PSNR 汇总")
    for tname, (_, p_vals, _) in all_results.items():
        if p_vals:
            print(f"  {tname:<10}  final={p_vals[-1]:.2f} dB  best={max(p_vals):.2f} dB")
    print(f"{'=' * 60}")
    print(f"\n结果已保存到: {os.path.abspath(args.save_dir)}\n")

    summary_path = os.path.join(args.save_dir, 'holographic_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(
            f"n_iter={args.n_iter}, M={args.M}, K={args.K}, "
            f"sigma={args.sigma}, sigma_final={args.sigma_final}, "
            f"lr={args.lr}, epsilon={args.epsilon}, "
            f"pixel_grouping={args.pixel_grouping}, "
            f"diffuser={not args.disable_diffuser}\n\n"
        )
        for tname, (_, p_vals, _) in all_results.items():
            if p_vals:
                f.write(f"{tname}: final={p_vals[-1]:.2f} dB, best={max(p_vals):.2f} dB\n")


if __name__ == '__main__':
    main()


### python train_holographic_ppo_yuzhe.py --targets grating,boat --n_iter 2000 --M 64 --K 4 --pixel_grouping 4 --sigma 0.15 --sigma_final 0.05 --lr 0.2
