"""
ppo.py  —  Gaussian policy + PPO trainer for the optical focusing task.

Policy:
    π_θ(φ) = N(μ, σ²·I)
    μ is trainable (at coarse resolution, then upsampled).
    σ is a fixed scalar.

Pixel grouping (g):
    In simulation with g=1 (per-pixel) and σ=0.04, the per-pixel gradient
    SNR ≈ √M/√(H·W) is too small for convergence.  Setting g>1 reduces the
    number of degrees of freedom from H·W to (H/g)·(W/g), raising SNR by g.
    With 128×128 SLM, g=4 → 32×32 = 1024 DOF,  SNR ≈ √(32/1024) ≈ 0.18.

PPO objective (paper Eq. 1):
    J(θ) = E[ min(r·A', clip(r, 1−ε, 1+ε)·A') ]
    r = π_θ(φ) / π_θ_old(φ),   A' = normalized advantage.
    Minimise −J.
"""

import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Gaussian policy
# ---------------------------------------------------------------------------

class GaussianPolicy(nn.Module): #定义了如何生成相位掩模的高斯策略类，包含了相位掩模的参数化方式（均值 μ 可训练，固定标准差 σ），以及像素分组大小（pixel_grouping）以控制训练参数数量和梯度 SNR。
    """
    Gaussian policy over SLM macro-pixels.

    Args:
        H, W           : full SLM spatial dimensions
        sigma          : fixed std [rad]
        pixel_grouping : g ≥ 1.  H and W must be divisible by g.
        device         : torch device
    """

    def __init__(self, H: int, W: int, sigma: float = 0.3,
                 pixel_grouping: int = 4, device: str = 'cpu'): 
        super().__init__()
        assert H % pixel_grouping == 0 and W % pixel_grouping == 0, \
            f"H={H}, W={W} must be divisible by pixel_grouping={pixel_grouping}"

        self.H, self.W = H, W
        self.sigma = sigma
        self.g     = pixel_grouping
        self.Hc    = H // pixel_grouping 
        self.Wc    = W // pixel_grouping
        self.device = device

        # Trainable mean at coarse resolution, initialized to zeros
        self.mu = nn.Parameter(torch.zeros(self.Hc, self.Wc, device=device)) #定义了可训练的相位掩模均值 μ，初始值为零张量，尺寸为 (Hc, Wc)，其中 Hc 和 Wc 是根据像素分组大小计算得到的粗略分辨率。这些参数将通过 PPO 训练进行优化，以学习出能够实现能量聚焦的最佳相位模式。

    # ------------------------------------------------------------------

    def _upsample(self, coarse: torch.Tensor) -> torch.Tensor:#用于将粗略分辨率的相位掩模均值 μ 从 (Hc, Wc) 上采样到全分辨率 (H, W)。这里使用了最近邻重复（nearest-neighbour repeat）的方式进行上采样，即将每个粗略像素的值复制到对应的 g×g 的块中，从而得到全分辨率的相位图。这种上采样方式简单且不会引入额外的平滑效果，适合于相位掩模的生成。
        """(*, Hc, Wc) → (*, H, W) via nearest-neighbour repeat."""
        if self.g == 1:
            return coarse
        return coarse.repeat_interleave(self.g, dim=-2) \
                     .repeat_interleave(self.g, dim=-1)

    def _downsample(self, fine: torch.Tensor) -> torch.Tensor:#用于将全分辨率的相位掩模样本 phi 从 (H, W) 下采样到粗略分辨率 (Hc, Wc)。这里通过取每个 g×g 块的左上角像素值来实现下采样，即从全分辨率的相位图中提取出对应于粗略分辨率的像素值。这种下采样方式与上采样方式相对应，确保了在计算 log-probability 时能够正确地将全分辨率的相位图映射到粗略分辨率的参数空间。
        """(M, H, W) → (M, Hc, Wc) by top-left corner of each block."""
        if self.g == 1:
            return fine
        return fine[:, ::self.g, ::self.g]

    # ------------------------------------------------------------------

    def sample(self, M: int) -> torch.Tensor:#根据当前策略参数（均值 μ 和固定标准差 σ）生成 M 个相位掩模样本。首先在粗略分辨率 (Hc, Wc) 上采样 M 个相位图，每个像素的值由对应的 μ 加上 σ 乘以一个标准正态随机数生成。然后将这些粗略分辨率的相位图通过 _upsample 方法上采样到全分辨率 (H, W)，得到最终的相位掩模样本。这些样本将被用来进行仿真测量和 PPO 更新。
        """
        Draw M phase masks from π_θ (detached — treated as constants).

        Returns:
            phi : tensor (M, H, W)
        """
        eps = torch.randn(M, self.Hc, self.Wc, device=self.device)
        phi_coarse = self.mu.detach() + self.sigma * eps  # (M, Hc, Wc)
        return self._upsample(phi_coarse)                  # (M, H, W)

    def log_prob(self, phi: torch.Tensor) -> torch.Tensor:#计算给定相位掩模样本 phi 在当前策略下的对数概率（log π_θ(φ)）。首先将输入的全分辨率相位图 phi 下采样到粗略分辨率 (Hc, Wc)，然后根据高斯分布的概率密度函数计算每个样本的 log-probability。由于 σ 是固定的，且每个像素独立同分布，log-probability 可以简化为每个像素差值平方的平均值除以 2σ² 的负数。这个 log-probability 将被用来计算 PPO 的概率比 r 和损失函数。
        """
        log π_θ(φ) for each sample (normalization constant omitted).

        log π ∝ −mean_k[(φ_k − μ_k)²] / (2σ²)

        Args:
            phi : tensor (M, H, W)

        Returns:
            lp : tensor (M,)
        """
        phi_c  = self._downsample(phi)               # (M, Hc, Wc)
        diff_sq = (phi_c - self.mu) ** 2             # (M, Hc, Wc)
        return -diff_sq.mean(dim=(-2, -1)) / (2.0 * self.sigma ** 2)#计算每个样本的 log-probability，结果是一个形状为 (M,) 的张量，其中每个元素对应一个相位掩模样本的 log-probability。这个值将被用来计算 PPO 的概率比 r 和损失函数，从而指导策略的更新。

    def get_mean(self) -> torch.Tensor:#获取当前策略的相位掩模均值 μ，并将其上采样到全分辨率 (H, W)。这个方法通常在评估阶段使用，用于生成最终的相位图进行仿真测量和性能评估。返回的相位图是一个形状为 (H, W) 的张量，表示当前策略下的最佳相位模式。
        """Return the current policy mean upsampled to (H, W), detached."""
        mu_c = self.mu.detach().clone().unsqueeze(0)  # (1, Hc, Wc)
        return self._upsample(mu_c).squeeze(0)         # (H, W)


# ---------------------------------------------------------------------------
# PPO trainer
# ---------------------------------------------------------------------------

class PPOTrainer:#
    """
    PPO trainer for the Gaussian optical policy.

    Args:
        policy  : GaussianPolicy
        lr      : Adam learning rate
        epsilon : PPO clip parameter ε
        K       : number of inner update steps per batch
    """

    def __init__(self, policy: GaussianPolicy, lr: float = 0.3,
                 epsilon: float = 0.02, K: int = 4):
        self.policy   = policy
        self.epsilon  = epsilon
        self.K        = K 
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)

    # ------------------------------------------------------------------

    def update(self, phi_samples: torch.Tensor, 
               losses: torch.Tensor) -> dict:
        """
        Perform K PPO gradient steps.

        Args:
            phi_samples : (M, H, W) — sampled masks (constants)
            losses      : (M,)       — measured task losses

        Returns:
            info : dict with 'mean_loss' and 'ppo_loss_last'
        """
        device = self.policy.device

        phi_samples = phi_samples.detach().to(device)
        losses      = losses.float().to(device)

        # Advantages: A = −L,  normalized
        adv      = -losses#将损失值转换为优势值 A，优势值的定义是 A = -L，即损失越小优势越大。这个转换是因为 PPO 的目标是最大化优势值，而不是最小化损失值。
        adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv_norm = adv_norm.detach()

        # Old log-probabilities (fixed reference for ratio)
        with torch.no_grad():
            log_prob_old = self.policy.log_prob(phi_samples)  #计算采样的相位掩模样本 phi_samples 在当前策略下的 log-probability，作为 PPO 更新中的固定参考值。这个 log-probability 将被用来计算概率比 r = π_θ(φ) / π_θ_old(φ)，其中 π_θ_old 是在采样时策略的概率分布。通过 detach() 方法将 log_prob_old 从计算图中分离出来，确保在后续的 PPO 更新中它被视为常数，不会对梯度计算产生影响。

        ppo_loss_last = None
        for _ in range(self.K):
            self.optimizer.zero_grad()

            log_prob_new = self.policy.log_prob(phi_samples)
            log_ratio    = (log_prob_new - log_prob_old).clamp(-5.0, 5.0)#裁剪，如果过大就舍去，避免数值不稳定
            r            = torch.exp(log_ratio)#计算概率比 r = π_θ(φ) / π_θ_old(φ)，其中 π_θ(φ) 是当前策略下的概率密度，π_θ_old(φ) 是采样时策略下的概率密度。通过对 log-probability 的差值进行指数运算得到概率比 r，这个值将被用来计算 PPO 的损失函数，从而指导策略的更新。通过 clamp(-5.0, 5.0) 来限制 log_ratio 的范围，避免在计算 r 时出现数值不稳定的问题。

            surr1 = r * adv_norm#计算 PPO 的第一个损失项 surr1 = r * A'，其中 r 是概率比，A' 是归一化的优势值。这个项表示了在当前策略下，相对于旧策略的优势值乘以概率比的结果，是 PPO 损失函数中的核心部分。
            surr2 = r.clamp(1.0 - self.epsilon, 1.0 + self.epsilon) * adv_norm#计算 PPO 的第二个损失项 surr2 = clip(r, 1−ε, 1+ε) * A'，其中 clip(r, 1−ε, 1+ε) 表示将概率比 r 限制在 [1−ε, 1+ε] 的范围内。这个项用于实现 PPO 的剪切机制，防止策略更新过大导致性能崩溃。通过比较 surr1 和 surr2，PPO 的损失函数能够在保证策略更新有效的同时，限制更新的幅度，从而提高训练的稳定性。
            ppo_loss = -torch.min(surr1, surr2).mean()

            ppo_loss.backward()
            self.optimizer.step() 
            ppo_loss_last = ppo_loss.item()

        return {
            'mean_loss'    : losses.mean().item(),
            'ppo_loss_last': ppo_loss_last,
        }
