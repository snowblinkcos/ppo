"""
train_mnist_ppo.py
==================
Replicate the MNIST optical image classification experiment (Fig. 7) from:

  "Model-free optical processors using in situ reinforcement learning
   with proximal policy optimization"  Li et al., Light: Sci. & Appl. (2026)

Physical setup (Fig. 7a):
  - Single diffractive SLM layer trained to classify handwritten digits
  - Input: MNIST digit phase-encoded onto the SLM plane
      φ_total(x,y) = φ_digit(x,y) + φ_layer(x,y)
      where φ_digit = pixel_value × π,  φ_layer is the trainable PPO parameter
  - Output: detector plane with 10 class-specific regions (3-4-3 grid)
  - Prediction: class = argmax over detector region intensities

Training (PPO):
  - M phase masks sampled per iteration from Gaussian policy N(μ, σ²I)
  - All M masks evaluated on the SAME MNIST mini-batch (fair comparison)
  - Loss per mask = cross-entropy(log(ROI_scores), true_labels), averaged over batch
  - K inner PPO update steps reuse the same (mask, loss) pairs
  - Paper result: ~80% test accuracy with single diffractive layer (Fig. 2b)

Simulation note:
  - Paper uses 800×800 SLM; this script uses SLM_SIZE×SLM_SIZE (default 200)
    for tractable CPU/GPU computation.  Scale pixel_size accordingly so the
    physical aperture and Fresnel number are preserved.

Usage:
  python train_mnist_ppo.py
  python train_mnist_ppo.py --n_iter 500 --M 32 --slm_size 200 --save_dir ./results_mnist
  python train_mnist_ppo.py --slm_size 128 --pixel_grouping 4   # faster
"""

import os
import argparse
import math

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from optical_sim import asm_propagate, make_grid_rois, roi_scores
from ppo import GaussianPolicy, PPOTrainer


# ---------------------------------------------------------------------------
# Default physical constants  (MNIST classification, Fig. 7)
# ---------------------------------------------------------------------------

WAVELENGTH  = 532e-9    # 532 nm green laser
Z           = 0.20      # 20 cm propagation (SLM → detector plane)
PIXEL_SIZE  = 8e-6      # 8 μm pixel pitch  →  200 px × 8 μm = 1.6 mm aperture
SLM_SIZE    = 200       # 200×200 pixels  (paper: 800×800, scaled for speed)
ROI_SIZE    = 25        # side length of each square detector ROI [pixels]
N_CLASSES   = 10        # MNIST digit classes 0-9


# ---------------------------------------------------------------------------
# MNIST diffractive classification task
# ---------------------------------------------------------------------------

class MNISTClassificationTask:
    """
    MNIST digit classification via a single trainable diffractive layer.

    Physical model (Fig. 7a):
      1. MNIST digit image  →  phase-encoded onto SLM plane
             φ_digit = pixel_value × π   (maps [0,1] → [0, π])
      2. Trainable diffractive layer adds its phase on the same plane:
             φ_total = φ_digit + φ_layer
      3. Input field at SLM:  E_in = exp(i·φ_total)
      4. ASM propagation to detector plane
      5. Detected intensity → sum over 10 class-specific ROIs
      6. Loss = cross_entropy(log(ROI_scores), true_label)

    Args:
        device   : torch device string
        slm_size : SLM/detector plane spatial resolution (pixels)
        roi_size : side length of each square detector region (pixels)
        data_dir : where to download/find the MNIST dataset
        batch_size : number of MNIST images per policy evaluation
    """

    def __init__(self, device: str, slm_size: int = SLM_SIZE,
                 roi_size: int = ROI_SIZE, data_dir: str = './data',
                 batch_size: int = 64):
        self.device     = device
        self.slm_size   = slm_size
        self.batch_size = batch_size

        # 10 class-specific detector ROIs in 3-4-3 grid (Fig. 7a)
        self.rois = make_grid_rois(slm_size, slm_size, roi_size=roi_size) # returns list of (y0, y1, x0, x1) for each ROI
        assert len(self.rois) == N_CLASSES, \
            f"make_grid_rois returned {len(self.rois)} ROIs, expected {N_CLASSES}"

        # ── MNIST dataset ──────────────────────────────────────────────────
        transform = transforms.Compose([
            transforms.Resize((slm_size, slm_size)),
            transforms.ToTensor(),          # values in [0, 1]
        ])
        train_ds = torchvision.datasets.MNIST(
            data_dir, train=True,  download=True, transform=transform)
        test_ds  = torchvision.datasets.MNIST(
            data_dir, train=False, download=True, transform=transform)

        self._train_loader_cfg = dict(
            dataset=train_ds, batch_size=batch_size,
            shuffle=True, num_workers=0, drop_last=True)
        self._train_iter = iter(
            torch.utils.data.DataLoader(**self._train_loader_cfg))##双星号表示将字典中的键值对作为关键字参数传递给函数

        self.test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=256, shuffle=False, num_workers=0)

        print(f"  MNIST: {len(train_ds)} train / {len(test_ds)} test images")
        print(f"  Detector ROIs (3-4-3 grid): {len(self.rois)} regions, "
              f"roi_size={roi_size}px")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_batch(self):
        """Return next mini-batch, resetting the loader when exhausted."""
        try:
            imgs, lbls = next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(
                torch.utils.data.DataLoader(**self._train_loader_cfg))
            imgs, lbls = next(self._train_iter)
        return imgs.to(self.device), lbls.to(self.device)

    @staticmethod
    def _phase_encode(images: torch.Tensor) -> torch.Tensor:
        """
        Phase-encode MNIST images.

        pixel ∈ [0, 1]  →  phase φ = pixel × π ∈ [0, π]

        Args:
            images : float (B, 1, H, W)
        Returns:
            phi    : float (B, H, W)
        """
        return images.squeeze(1) * math.pi   # (B, H, W)

    @torch.no_grad()
    def _propagate(self, phi_layer: torch.Tensor,
                   phi_digit: torch.Tensor) -> torch.Tensor:
        """
        Simulate one diffractive layer measurement.

        φ_total = φ_digit + φ_layer  →  E = exp(i·φ_total)  →  ASM  →  |E|²

        Args:
            phi_layer : (H, W)   trainable diffractive layer phase
            phi_digit : (B, H, W) phase-encoded MNIST images
        Returns:
            intensity : (B, H, W)
        """
        phi_total = phi_digit + phi_layer.unsqueeze(0)          # (B, H, W)
        field = torch.exp(1j * phi_total.float()).to(torch.complex64)
        out = asm_propagate(field, Z, WAVELENGTH, PIXEL_SIZE)
        return out.abs() ** 2                                   # (B, H, W)

    # ------------------------------------------------------------------
    # Public interface used by the PPO training loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_losses(self, phi_samples: torch.Tensor) -> torch.Tensor:
        """
        Evaluate M phase masks on one mini-batch and return cross-entropy loss.

        All M masks share the SAME mini-batch so their losses are comparable,
        matching the paper's in-situ measurement protocol.

        Args:
            phi_samples : (M, H, W) — sampled diffractive layer masks

        Returns:
            losses : (M,) on CPU
        """
        images, labels = self._next_batch()          # same batch for all M masks
        phi_digit = self._phase_encode(images)       # (B, H, W)

        M = phi_samples.shape[0]
        losses = torch.zeros(M)

        for m in range(M):
            phi_layer = phi_samples[m].to(self.device)
            intensity = self._propagate(phi_layer, phi_digit)   # (B, H, W)
            scores    = roi_scores(intensity, self.rois)         # (B, 10)
            log_scores = torch.log(scores + 1e-12)
            losses[m]  = F.cross_entropy(log_scores, labels, reduction='mean').item()

        return losses   # (M,) on CPU

    @torch.no_grad()
    def evaluate_accuracy(self, phi_layer: torch.Tensor) -> float:
        """
        Compute classification accuracy on the full MNIST test set.

        Uses the deterministic policy mean (no sampling noise).

        Args:
            phi_layer : (H, W) — learned diffractive layer phase (policy mean)
        Returns:
            accuracy  : float in [0, 1]
        """
        phi_layer = phi_layer.to(self.device)
        correct, total = 0, 0

        for images, labels in self.test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            phi_digit = self._phase_encode(images)              # (B, H, W)
            intensity = self._propagate(phi_layer, phi_digit)   # (B, H, W)
            scores    = roi_scores(intensity, self.rois)         # (B, 10)
            preds     = scores.argmax(dim=1)
            correct  += (preds == labels).sum().item()
            total    += labels.size(0)

        return correct / total


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _save_phase(phase: torch.Tensor, path: str, title: str):
    arr = phase.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(arr % (2 * math.pi), cmap='hsv', vmin=0, vmax=2 * math.pi,
                   interpolation='nearest')
    fig.colorbar(im, ax=ax).set_label('Phase (rad)')
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _save_detector(intensity: torch.Tensor, rois: list, label: int,
                   path: str, title: str):
    """Grayscale detector image with ROI borders; correct ROI highlighted."""
    arr = intensity.detach().cpu().numpy().astype(np.float32)
    arr = arr / (arr.max() + 1e-12)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(arr, cmap='gray', vmin=0, vmax=1,
              interpolation='nearest', aspect='equal')
    for i, (y0, y1, x0, x1) in enumerate(rois):
        color = 'red'    if i == label else 'white'
        lw    = 2.0      if i == label else 0.8
        rect  = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                               linewidth=lw, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    plt.tight_layout(pad=0.5)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _save_convergence(iters: list, acc_vals: list, path: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iters, [v * 100 for v in acc_vals],
            linewidth=2, color='tab:orange', label='PPO')
    ax.axhline(10.0, linestyle='--', color='gray', label='random (10%)')
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('MNIST Optical Classification – PPO (Fig. 7b)')
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _save_class_examples(task: MNISTClassificationTask,
                         phi_layer: torch.Tensor,
                         save_dir: str):
    """
    For each digit class, show: input phase | detector intensity | class scores.
    Reproduces the visual style of Fig. 7c.
    """
    device = task.device
    phi_layer = phi_layer.to(device)

    # Grab one test example per class
    examples = {}
    for images, labels in task.test_loader:
        for img, lbl in zip(images, labels):
            c = lbl.item()
            if c not in examples:
                examples[c] = img
            if len(examples) == N_CLASSES:
                break
        if len(examples) == N_CLASSES:
            break

    fig, axes = plt.subplots(N_CLASSES, 3, figsize=(9, 3 * N_CLASSES))
    col_titles = ['Phase-encoded input', 'Detector intensity', 'Class score']
    for col, ct in enumerate(col_titles):
        axes[0, col].set_title(ct, fontsize=10, fontweight='bold')

    for c in range(N_CLASSES):
        img = examples[c].to(device)            # (1, H, W)
        phi_digit = task._phase_encode(img.unsqueeze(0))   # (1, H, W)
        intensity = task._propagate(phi_layer, phi_digit)  # (1, H, W)
        scores    = roi_scores(intensity, task.rois)[0]    # (10,)

        # Column 0: phase-encoded input
        ax0 = axes[c, 0]
        ax0.imshow(phi_digit[0].cpu().numpy(), cmap='twilight',
                   vmin=0, vmax=math.pi)
        ax0.set_ylabel(f'Digit {c}', fontsize=9)
        ax0.axis('off')

        # Column 1: detector intensity with ROI overlay
        ax1 = axes[c, 1]
        det = intensity[0].cpu().numpy()
        det = det / (det.max() + 1e-12)
        ax1.imshow(det, cmap='gray', vmin=0, vmax=1)
        for i, (y0, y1, x0, x1) in enumerate(task.rois):
            color = 'red' if i == c else 'white'
            lw    = 1.8   if i == c else 0.6
            rect  = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   linewidth=lw, edgecolor=color,
                                   facecolor='none')
            ax1.add_patch(rect)
        ax1.axis('off')

        # Column 2: horizontal bar chart of class scores
        ax2 = axes[c, 2]
        sc = scores.cpu().numpy()
        sc = sc / (sc.sum() + 1e-12)
        colors = ['tab:red' if i == c else 'tab:green' for i in range(N_CLASSES)]
        ax2.barh(range(N_CLASSES), sc, color=colors)
        ax2.set_xlim(0, 1)
        ax2.set_yticks(range(N_CLASSES))
        ax2.set_yticklabels([str(i) for i in range(N_CLASSES)], fontsize=7)
        ax2.invert_yaxis()
        ax2.set_xlabel('Score', fontsize=8)

    plt.suptitle('Fig. 7c – PPO Learned Diffractive Classifier', fontsize=12,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mnist_ppo_class_examples.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved class examples figure.")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='PPO MNIST optical image classification (Fig. 7)')

    # Physical setup
    p.add_argument('--slm_size',       type=int,   default=SLM_SIZE,
                   help='SLM spatial resolution in pixels (default: 200)')
    p.add_argument('--roi_size',       type=int,   default=ROI_SIZE,
                   help='Detector ROI side length in pixels (default: 25)')

    # PPO / policy
    p.add_argument('--n_iter',         type=int,   default=500,
                   help='Number of PPO training iterations')
    p.add_argument('--M',              type=int,   default=32,
                   help='Phase masks sampled per iteration')
    p.add_argument('--K',              type=int,   default=4,
                   help='PPO inner update steps per batch')
    p.add_argument('--sigma',          type=float, default=0.5,
                   help='Gaussian policy std σ [rad]')
    p.add_argument('--lr',             type=float, default=0.05,
                   help='Adam learning rate')
    p.add_argument('--epsilon',        type=float, default=0.02,
                   help='PPO clip parameter ε')
    p.add_argument('--pixel_grouping', type=int,   default=1,
                   help='Macro-pixel size g; DOF = (slm_size/g)²')

    # Training logistics
    p.add_argument('--batch_size',     type=int,   default=64,
                   help='MNIST images per mask evaluation')
    p.add_argument('--eval_every',     type=int,   default=25,
                   help='Evaluate test accuracy every N iterations')
    p.add_argument('--data_dir',       type=str,   default='./data')
    p.add_argument('--save_dir',       type=str,   default='./results_mnist')
    p.add_argument('--seed',           type=int,   default=42)
    p.add_argument('--device',         type=str,
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
    os.makedirs(args.data_dir, exist_ok=True)

    device = args.device
    if device == 'cuda':
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU.")
            device = 'cpu'
        else:
            # Quick sanity-check: some CUDA builds are incompatible with the GPU
            try:
                torch.zeros(1, device='cuda')
            except Exception as e:
                print(f"CUDA unusable ({e}), falling back to CPU.")
                device = 'cpu'

    dof = (args.slm_size // args.pixel_grouping) ** 2

    print(f"\n{'='*65}")
    print(f"  Experiment : MNIST Optical Classification (Fig. 7, PPO)")
    print(f"  SLM size   : {args.slm_size}×{args.slm_size} px  "
          f"(aperture = {args.slm_size * PIXEL_SIZE * 1e3:.1f} mm)")
    print(f"  pixel_size = {PIXEL_SIZE*1e6:.0f} μm   λ = {WAVELENGTH*1e9:.0f} nm   "
          f"z = {Z*100:.0f} cm")
    print(f"  pixel_grouping = {args.pixel_grouping}  →  DOF = {dof}")
    print(f"  n_iter={args.n_iter}  M={args.M}  K={args.K}  "
          f"batch_size={args.batch_size}")
    print(f"  σ={args.sigma}  lr={args.lr}  ε={args.epsilon}")
    print(f"  device = {device}")
    print(f"{'='*65}\n")

    # ── Build task, policy, trainer ───────────────────────────────────────
    task = MNISTClassificationTask(
        device=device,
        slm_size=args.slm_size,
        roi_size=args.roi_size,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )

    policy = GaussianPolicy(
        H=args.slm_size, W=args.slm_size,
        sigma=args.sigma,
        pixel_grouping=args.pixel_grouping,
        device=device,
    )

    trainer = PPOTrainer(policy, lr=args.lr, epsilon=args.epsilon, K=args.K)

    # ── Training loop ─────────────────────────────────────────────────────
    eval_iters = []
    eval_acc   = []
    best_acc   = 0.0
    best_phase = None

    print(f"Starting PPO training for {args.n_iter} iterations...\n")

    for it in range(1, args.n_iter + 1):

        # 1. Sample M diffractive layer phase masks from current policy
        phi_samples = policy.sample(args.M)               # (M, H, W)

        # 2. Evaluate each mask on the same MNIST mini-batch
        losses = task.compute_losses(phi_samples)          # (M,) CPU

        # 3. PPO update (K inner steps reusing the same batch)
        info = trainer.update(phi_samples, losses)

        # 4. Periodic evaluation on full test set
        if it % args.eval_every == 0 or it == 1:
            mean_phase = policy.get_mean()                 # (H, W)
            acc = task.evaluate_accuracy(mean_phase)
            eval_iters.append(it)
            eval_acc.append(acc)

            if acc > best_acc:
                best_acc   = acc
                best_phase = mean_phase.clone()

            print(f"[{it:5d}/{args.n_iter}]  "
                  f"test_acc={acc*100:.2f}%  "
                  f"mean_loss={info['mean_loss']:.4f}  "
                  f"ppo_loss={info['ppo_loss_last']:.4f}")

    print(f"\nBest test accuracy: {best_acc*100:.2f}%")

    # ── Save results ──────────────────────────────────────────────────────
    final_phase = policy.get_mean()   # (H, W)

    # Convergence curve  (Fig. 7b style)
    _save_convergence(
        eval_iters, eval_acc,
        path=os.path.join(args.save_dir, 'mnist_ppo_convergence.png'),
    )

    # Final learned phase pattern  (Fig. 7c style, phase panel)
    _save_phase(
        final_phase,
        path=os.path.join(args.save_dir, 'mnist_ppo_final_phase.png'),
        title=f'Learned Diffractive Layer – PPO  (acc={best_acc*100:.1f}%)',
    )

    # Classification examples for all 10 digits  (Fig. 7c)
    _save_class_examples(task, final_phase, args.save_dir)

    # Save best phase tensor for further analysis
    if best_phase is not None:
        torch.save(
            best_phase.cpu(),
            os.path.join(args.save_dir, 'mnist_ppo_best_phase.pt'),
        )

    print(f"\nResults saved → {os.path.abspath(args.save_dir)}/")
    print("  mnist_ppo_convergence.png  — test accuracy vs iteration (Fig. 7b)")
    print("  mnist_ppo_final_phase.png  — learned diffractive layer phase")
    print("  mnist_ppo_class_examples.png — per-class classification results (Fig. 7c)")
    print("  mnist_ppo_best_phase.pt    — best phase mask tensor")


if __name__ == '__main__':
    main()
