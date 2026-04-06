"""
train_energy_focusing_ppo.py
============================
Replicate the **Energy Focusing** experiment from:

  "Model-free optical processors using in situ reinforcement learning
   with proximal policy optimization"  Li et al., Light: Sci. & Appl. (2026)

Physical noise model (applied during training measurements):
  1. SLM phase quantization  — 8-bit SLM can only apply 256 discrete levels
  2. SLM phase noise         — per-pixel Gaussian noise (drift, vibration, calibration)
  3. Shot noise              — camera photon-counting Poisson noise

Evaluation (energy_ratio) uses the clean mean phase μ without added noise,
representing the best achievable ER for the learned phase pattern.

Usage
-----
  python train_energy_focusing_ppo.py
  python train_energy_focusing_ppo.py --n_iter 2000 --save_dir ./results
  # Paper physical params (slow convergence, add noise):
  python train_energy_focusing_ppo.py --sigma 0.04 --pixel_grouping 1 --M 256
"""

import os
import argparse
import math

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from optical_sim import asm_propagate, make_grid_rois, roi_scores
from ppo import GaussianPolicy, PPOTrainer


# ---------------------------------------------------------------------------
# Physical / task constants (from paper)
# ---------------------------------------------------------------------------

WAVELENGTH  = 520e-9   # 520 nm
Z           = 9.6e-2   # 9.6 cm
PIXEL_SIZE  = 16e-6    # 16 μm  (focusing layer)
LAYER_SIZE  = 128      # 128×128 pixels
N_STRIPS    = 10       # 10 horizontal detector strips


# ---------------------------------------------------------------------------
# Physical noise model
# ---------------------------------------------------------------------------

class PhysicsNoise:
    """
    Simulates hardware imperfections present in a real SLM + camera system.

    Applied during training measurements so the PPO gradient estimate is
    degraded by realistic noise — just as in the physical experiment.

    Three sources:
      1. Phase quantization : 8-bit SLM → 256 discrete levels, step = 2π/256     相位量化：8位SLM产生256个离散相位级，步长为2π/256
      2. SLM phase noise    : per-pixel Gaussian additive noise σ_slm [rad]      
                              (models drift, vibration, calibration error)       SLM相位噪声：每个像素的高斯加性噪声，模拟漂移、振动、校准误差
      3. Shot noise         : Poisson noise on detected photon counts
                              (scale intensity to peak_photons, then sample)     散粒噪声：探测光子计数上的泊松噪声

    Args:
        n_bits        : SLM bit-depth (default 8 → 256 levels; 0 = disabled)
        slm_phase_std : SLM phase noise std-dev [rad] (default 0.1)
        peak_photons  : peak photon count per pixel (default 200; 0 = disabled)
    """

    def __init__(self, n_bits: int = 8, slm_phase_std: float = 0.1,
                 peak_photons: int = 200):
        self.n_bits        = n_bits
        self.slm_phase_std = slm_phase_std
        self.peak_photons  = peak_photons

    def apply_to_phase(self, phase: torch.Tensor) -> torch.Tensor:
        """
        Apply quantization + additive phase noise to SLM phase masks.

        Args:
            phase : float (M, H, W)

        Returns:
            noisy_phase : float (M, H, W)
        """
        # 1. Phase quantization
        if self.n_bits > 0:
            levels = 2 ** self.n_bits               # e.g. 256
            step   = 2.0 * math.pi / levels         # ~0.0245 rad per step
            phase  = torch.round(phase / step) * step

        # 2. SLM phase noise (per-pixel, per-measurement)
        if self.slm_phase_std > 0:
            phase = phase + self.slm_phase_std * torch.randn_like(phase)

        return phase

    def apply_to_intensity(self, intensity: torch.Tensor) -> torch.Tensor:
        """
        Apply shot noise (Poisson) to detected intensity.

        Args:
            intensity : float (M, H, W), values ≥ 0

        Returns:
            noisy_intensity : float (M, H, W)
        """
        if self.peak_photons <= 0:
            return intensity

        # Scale so that the spatial maximum maps to peak_photons
        max_val = intensity.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
        photons = (intensity / max_val) * self.peak_photons   # (M, H, W)

        # Poisson sampling — torch.poisson requires float input ≥ 0
        noisy_photons = torch.poisson(photons.clamp(min=0.0))

        # Convert back to intensity units
        return (noisy_photons / self.peak_photons) * max_val


# ---------------------------------------------------------------------------
# Task: energy focusing
# ---------------------------------------------------------------------------

class FocusingTask:
    """
    Plane wave → SLM phase mask → ASM propagation → 10 horizontal strips.

    Goal  : maximise energy in the target strip.
    Loss  : CrossEntropy(log(strip_scores), target)
    Metric: ER = score[target] / Σ score[k]

    During training, physical noise is injected into each measurement.
    Evaluation uses the clean (noise-free) mean phase μ.
    """

    def __init__(self, target_strip: int, device: str,
                 noise: PhysicsNoise = None):
        self.target_strip = target_strip
        self.device       = device
        self.noise        = noise   # None → ideal simulation

        self.plane_wave = torch.ones(
            LAYER_SIZE, LAYER_SIZE, dtype=torch.complex64, device=device
        )##创建一个平面波
        self.rois    = make_grid_rois(LAYER_SIZE, LAYER_SIZE, roi_size=20)##生成一个ROI区域
        self._target = torch.tensor([target_strip], dtype=torch.long, device=device)

    # ------------------------------------------------------------------

    def _propagate(self, phase_masks: torch.Tensor,
                   add_noise: bool = False) -> torch.Tensor:  ##执行光传播模拟
        """
        Args:
            phase_masks : float (M, H, W)
            add_noise   : whether to apply PhysicsNoise

        Returns:
            intensities : float (M, H, W)
        """
        phi = phase_masks.to(self.device)

        # Apply physical noise to the SLM phase before propagation
        if add_noise and self.noise is not None:
            phi = self.noise.apply_to_phase(phi)

        slm_fields = self.plane_wave.unsqueeze(0) * torch.exp(
            1j * phi.float()        ##通过SLM改变光场，用复数表示振幅和相位
        ).to(torch.complex64)                             # (M, H, W)

        out = asm_propagate(slm_fields, Z, WAVELENGTH, PIXEL_SIZE)  ##通过角谱法传播模拟光场传播
        intensity = out.abs() ** 2             ##物理意义：光电探测器测量的是光强，与电场振幅的平方成正比           
                                                # (M, H, W)
        # Apply shot noise to the detected intensity
        if add_noise and self.noise is not None:
            intensity = self.noise.apply_to_intensity(intensity)

        return intensity

    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_losses(self, phase_masks: torch.Tensor) -> torch.Tensor:
        """
        Cross-entropy loss for M phase masks WITH physical noise.

        Returns:
            losses : float (M,)  on CPU
        """
        intensities = self._propagate(phase_masks, add_noise=True)
        scores      = roi_scores(intensities, self.rois)             # (M, 10)
        log_scores  = torch.log(scores + 1e-12)
        target      = self._target.expand(phase_masks.shape[0])
        losses      = F.cross_entropy(log_scores, target, reduction='none')
        return losses.cpu()

    # ------------------------------------------------------------------

    @torch.no_grad()
    def energy_ratio(self, phase_mask: torch.Tensor) -> float:   ##评估能量比
        """
        Evaluate ER = score[target] / Σ score[k] with clean phase (no noise).
        Represents the true ER of the learned phase pattern.
        """
        intensities = self._propagate(phase_mask.unsqueeze(0), add_noise=False)
        scores = roi_scores(intensities, self.rois)          # (1, 10)
        return float(scores[0, self.target_strip] /
                     (scores[0].sum() + 1e-12))


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _save_phase(phase: torch.Tensor, path: str, title: str):##保存相位图
    arr = phase.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(arr, cmap='hsv', vmin=0, vmax=2 * math.pi,
                   interpolation='nearest')
    fig.colorbar(im, ax=ax).set_label('Phase (rad)')
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _save_intensity(intensity: torch.Tensor, path: str, title: str,
                    rois: list = None, target_strip: int = -1):##保存强度图
    """
    Grayscale camera-like image with 3-4-3 square ROI borders overlaid.
    Matches the visual style of the paper (Fig. 1a / 2d):
      - white rectangles for all 10 ROIs
      - yellow rectangle for the target ROI
    """
    arr = intensity.detach().cpu().numpy().astype(np.float32)
    arr = arr / (arr.max() + 1e-12)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(arr, cmap='gray', vmin=0, vmax=1,
              interpolation='nearest', aspect='equal')

    if rois is not None:
        for i, (y0, y1, x0, x1) in enumerate(rois):
            color = 'yellow' if i == target_strip else 'white'
            lw    = 2.0      if i == target_strip else 0.9
            rect  = plt.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                linewidth=lw, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)

    ax.set_title(title, fontsize=10, pad=4)
    ax.axis('off')
    plt.tight_layout(pad=0.5)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _save_convergence(iters, er_vals, path: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iters, [v * 100 for v in er_vals], linewidth=2, color='tab:blue')
    ax.axhline(100.0 / N_STRIPS, linestyle='--', color='gray',
               label=f'random baseline ({100/N_STRIPS:.1f}%)')
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Energy Ratio (%)')
    ax.set_title('Energy Focusing – PPO (with physical noise)')
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():    #定义命令行参数，设置训练配置和物理噪声模型
    p = argparse.ArgumentParser(
        description='PPO energy focusing with physical noise model')

    # Training
    p.add_argument('--n_iter',         type=int,   default=2000)#迭代次数
    p.add_argument('--M',              type=int,   default=256) #每次迭代采样的相位掩模数量（论文建议 M=32，但更大的 M=256 可以加速收敛，尤其是在有噪声的情况下）
    p.add_argument('--K',              type=int,   default=4)   #每次 PPO 更新的 epoch 数（论文建议 K=4，能量聚焦每个 mask 测一次）
    p.add_argument('--sigma',          type=float, default=0.3) #高斯策略的固定标准差 σ [rad]，控制探索程度。论文物理实验使用 σ=0.04，但在理想仿真中可以使用更大的 σ=0.3 加速训练（尤其是当 pixel_grouping=1 时）。如果收敛不稳定，可以尝试调大 σ。
    p.add_argument('--lr',             type=float, default=0.3) #PPO 优化器的学习率。论文物理实验使用 lr=0.3，但在理想仿真中可以使用更大的 lr=0.5 加速训练。如果收敛不稳定，可以尝试调大 lr。
    p.add_argument('--epsilon',        type=float, default=0.02)#PPO 的 clipping 参数 ε。论文物理实验使用 ε=0.02，但在理想仿真中可以使用更大的 ε=0.1 加速训练。如果收敛不稳定，可以尝试调大 ε。
    p.add_argument('--pixel_grouping', type=int,   default=1) #像素分组大小 g。论文物理实验使用 g=4（每 4×4 像素共享一个训练参数），以提高梯度 SNR 和加速收敛。在理想仿真中可以使用 g=1（每像素独立训练）以获得更高最终 ER，但收敛会更慢，尤其是在有噪声的情况下。如果收敛不稳定，可以尝试调大 pixel_grouping。
    p.add_argument('--target_strip',   type=int,   default=0) #能量聚焦的目标 ROI 条索引（0-9）。论文实验中使用 target_strip=0，但可以尝试其他索引以验证算法的普适性。
    p.add_argument('--eval_every',     type=int,   default=20)#评估间隔（每多少迭代评估一次 ER）。论文实验中使用 eval_interval=2，但在理想仿真中可以使用更大的 eval_every=20 以减少评估频率和加速训练。

    # Physical noise
    p.add_argument('--n_bits',         type=int,   default=8) #SLM 位深（默认 8 位 → 256 个离散相位级）。论文物理实验使用 8-bit SLM，但在理想仿真中可以设置 n_bits=0 以禁用量化噪声。如果收敛不稳定，可以尝试调大 n_bits。
    p.add_argument('--slm_noise',      type=float, default=0.75) #SLM 相位噪声标准差 σ_slm [rad]，模拟漂移、振动、校准误差。论文物理实验中 σ_slm 的具体值未给出，但可以通过调节 slm_noise 来模拟不同程度的系统不稳定性。如果收敛不稳定，可以尝试调大 slm_noise。
    p.add_argument('--peak_photons',   type=int,   default=50)# 散粒噪声的峰值光子数，控制探测强度上的泊松噪声水平。论文物理实验中 peak_photons 的具体值未给出，但可以通过调节 peak_photons 来模拟不同程度的散粒噪声。如果收敛不稳定，可以尝试调小 peak_photons。
    p.add_argument('--no_noise',       action='store_true') #是否禁用物理噪声，进行理想仿真。论文物理实验中包含噪声，但在理想仿真中可以使用 --no_noise 来禁用所有物理噪声，以验证算法在无噪声条件下的性能上限。

    p.add_argument('--save_dir',       type=str,   default='./results')#结果保存目录，包含最终相位图、强度图、收敛曲线，以及最佳 ER 的相位参数（best_energy_focusing_mu.pt）。论文中未指定保存路径，但在复现过程中建议定期保存结果以便分析和可视化。
    p.add_argument('--seed',           type=int,   default=42)  #随机种子，确保结果可复现。论文中未指定随机种子，但在复现过程中设置固定的 seed 可以帮助获得稳定的结果并便于调试。
    p.add_argument('--device',         type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')  
    return p.parse_args()#解析命令行参数，设置训练配置和物理噪声模型


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args() 

    torch.manual_seed(args.seed) #设置 PyTorch 随机种子，确保结果可复现
    np.random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)   #创建结果保存目录，如果不存在的话
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = 'cpu'
    print(f"Device: {device}")

    # Noise model
    noise = None if args.no_noise else PhysicsNoise(
        n_bits        = args.n_bits,
        slm_phase_std = args.slm_noise,
        peak_photons  = args.peak_photons,
    )

    dof = (LAYER_SIZE // args.pixel_grouping) ** 2  #有效训练参数数量（自由度），影响梯度 SNR 和收敛速度
    print(f"\n{'='*60}")
    print(f"  Task         : Energy Focusing  (strip {args.target_strip}/{N_STRIPS})")
    print(f"  n_iter={args.n_iter}  M={args.M}  K={args.K}")
    print(f"  sigma={args.sigma}  lr={args.lr}  epsilon={args.epsilon}")
    print(f"  pixel_grouping={args.pixel_grouping}  (DOF={dof})")
    if noise:
        print(f"  Noise: quant={args.n_bits}-bit | "
              f"SLM phase std={args.slm_noise} rad | "
              f"shot noise peak={args.peak_photons} photons")
    else:
        print(f"  Noise: DISABLED (ideal simulation)")
    print(f"{'='*60}\n")

    task    = FocusingTask(target_strip=args.target_strip,
                           device=device, noise=noise)  #创建能量聚焦任务实例，包含目标 ROI、物理噪声模型等配置
    policy  = GaussianPolicy(H=LAYER_SIZE, W=LAYER_SIZE,
                             sigma=args.sigma,
                             pixel_grouping=args.pixel_grouping,
                             device=device) #创建高斯策略实例，定义了相位掩模的参数化方式（均值 μ 可训练，固定标准差 σ），以及像素分组大小（pixel_grouping）以控制训练参数数量和梯度 SNR
    trainer = PPOTrainer(policy, lr=args.lr,
                         epsilon=args.epsilon, K=args.K) #创建 PPO 训练器实例，负责根据采样的相位掩模和对应的损失值进行策略更新，使用指定的学习率、剪切参数 ε 和 epoch 数 K

    eval_iters = [] #记录评估迭代次数的列表，用于后续绘制收敛曲线
    eval_er    = [] #记录评估能量比（ER）值的列表，用于后续绘制收敛曲线和分析算法性能
    best_er    = 0.0    
    best_phase = None

    for it in range(1, args.n_iter + 1):    
        phi_samples = policy.sample(args.M)                 # (M, H, W)
        losses      = task.compute_losses(phi_samples)     # (M,)  with noise
        info        = trainer.update(phi_samples, losses)   #执行 PPO 更新，根据采样的相位掩模和对应的损失值进行策略更新，返回包含平均损失和最后一次 PPO 损失的字典信息

        if it % args.eval_every == 0 or it == 1:
            mean_phase = policy.get_mean()                 # (H, W)
            er = task.energy_ratio(mean_phase)             # clean eval
            eval_iters.append(it) #记录当前评估迭代次数
            eval_er.append(er) 

            if er > best_er:
                best_er    = er
                best_phase = mean_phase.clone()

            print(f"[{it:4d}/{args.n_iter}]  ER={er*100:.2f}%  "
                  f"mean_loss={info['mean_loss']:.4f}  "
                  f"ppo_loss={info['ppo_loss_last']:.4f}")

    print(f"\nBest ER (clean eval): {best_er*100:.2f}%")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    final_phase = policy.get_mean() #获取最终训练完成的相位掩模均值 μ，作为最终的相位图进行评估和可视化。这个均值代表了 PPO 学习到的最佳相位模式，在理想条件下应该能够实现较高的能量聚焦效果。

    with torch.no_grad():
        slm = task.plane_wave * torch.exp(
            1j * final_phase.float().to(device)
        ).to(torch.complex64) #通过最终的相位掩模 μ 生成 SLM 上的复数光场，表示为振幅（平面波）和相位（最终训练得到的相位图）的乘积。这个光场将被传播模拟器用来计算最终的输出强度分布，以评估最终的能量聚焦效果。
        out = asm_propagate(slm.unsqueeze(0), Z, WAVELENGTH, PIXEL_SIZE) #使用角谱法传播模拟器对生成的 SLM 光场进行传播，得到传播后的输出光场。这个输出光场包含了最终训练得到的相位掩模在传播后形成的强度分布信息，可以用来计算最终的能量比（ER）和生成最终的强度图。
        final_intensity = out.abs()[0] ** 2 #计算最终的强度图，即输出光场的振幅平方。这个强度图表示了最终训练得到的相位掩模在传播后形成的光强分布，可以用来评估能量聚焦效果，并生成可视化图像。

    _save_phase(
        final_phase,
        path=os.path.join(args.save_dir, 'focusing_ppo_final_phase.png'),
        title=f'Final Phase – PPO  (ER={best_er*100:.1f}%)',
    )
    _save_intensity(
        final_intensity,
        path=os.path.join(args.save_dir, 'focusing_ppo_final_intensity.png'),#保存最终的强度图，文件名为 focusing_ppo_final_intensity.png，标题包含最终的能量比（ER）百分比，以便分析和展示 PPO 学习到的相位掩模在传播后形成的光强分布和能量聚焦效果。
        title=f'Final Intensity – PPO  ER={best_er*100:.1f}%',
        rois=task.rois,
        target_strip=args.target_strip,
    )
    _save_convergence(
        eval_iters, eval_er,
        path=os.path.join(args.save_dir, 'focusing_ppo_convergence.png'),
    )

    if best_phase is not None: 
        torch.save(best_phase.cpu(),
                   os.path.join(args.save_dir, 'best_energy_focusing_mu.pt'))
        print(f"Saved best phase → {args.save_dir}/best_energy_focusing_mu.pt")

    print(f"Results → {os.path.abspath(args.save_dir)}")


if __name__ == '__main__':
    main()
