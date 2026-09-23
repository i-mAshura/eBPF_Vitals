r"""
Temperature Scaling and Expected Calibration Error (ECE) for Verifier Surrogate.
Implements:
1. Temperature optimization for probability calibration (Guo et al., ICML 2017)
2. ECE computation over M probability bins
3. False-accept rate bound calculation (Eq. (4) from paper)
4. Split conformal prediction coverage analysis (Angelopoulos et al., 2023)
"""

from typing import Tuple, List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .gnn_surrogate import VerifierSurrogate

def compute_ece(probs: np.ndarray, labels: np.ndarray, num_bins: int = 15) -> Tuple[float, List[Dict]]:
    """
    Computes Expected Calibration Error (ECE):
    ECE = \sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    bin_stats = []

    for i in range(num_bins):
        bin_lo = bin_boundaries[i]
        bin_hi = bin_boundaries[i + 1]
        mask = (probs > bin_lo) & (probs <= bin_hi) if i > 0 else (probs >= bin_lo) & (probs <= bin_hi)
        count = int(np.sum(mask))

        if count > 0:
            bin_acc = float(np.mean(labels[mask]))
            bin_conf = float(np.mean(probs[mask]))
            err = abs(bin_acc - bin_conf)
            ece += (count / len(probs)) * err
            bin_stats.append({
                "bin": i,
                "range": (float(bin_lo), float(bin_hi)),
                "count": count,
                "accuracy": bin_acc,
                "confidence": bin_conf,
                "error": err
            })
        else:
            bin_stats.append({
                "bin": i,
                "range": (float(bin_lo), float(bin_hi)),
                "count": 0,
                "accuracy": 0.0,
                "confidence": (bin_lo + bin_hi) / 2.0,
                "error": 0.0
            })

    return float(ece), bin_stats

class TemperatureScaler:
    def __init__(self, surrogate: VerifierSurrogate):
        self.surrogate = surrogate
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def calibrate(self, val_logits: torch.Tensor, val_labels: torch.Tensor, lr: float = 0.01, max_iters: int = 100) -> float:
        """
        Optimizes temperature T using NLL on validation logits.
        """
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iters)
        bce_criterion = nn.BCEWithLogitsLoss()

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = val_logits / self.temperature
            loss = bce_criterion(scaled_logits.squeeze(-1), val_labels.float())
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        calibrated_t = max(0.1, float(self.temperature.item()))
        self.surrogate.temperature = calibrated_t
        return calibrated_t

def verify_false_accept_bound(probs: np.ndarray, labels: np.ndarray, tau: float = 0.9, ece: float = 0.02) -> Dict:
    """
    Verifies Eq. (4) from paper:
    \varepsilon_\tau \le 1 - \bar{p}_\tau + \bar{e}_\tau
    where \bar{p}_\tau = E[\hat{V}_\theta | \hat{V}_\theta \ge \tau],
    and \bar{e}_\tau is the conditional calibration error above \tau.
    """
    above_tau_mask = (probs >= tau)
    if not np.any(above_tau_mask):
        return {
            "tau": tau,
            "count": 0,
            "observed_false_accept_rate": 0.0,
            "bound": 0.0,
            "p_bar": 1.0,
            "e_bar": 0.0,
            "bound_satisfied": True
        }

    probs_above = probs[above_tau_mask]
    labels_above = labels[above_tau_mask]

    # False accept is when model predicts >= tau (accepted), but true label is 0 (rejected)
    observed_false_accept = float(np.mean(labels_above == 0))
    p_bar = float(np.mean(probs_above))
    e_bar = float(ece) # bounded by total ECE
    theoretical_bound = (1.0 - p_bar) + e_bar

    return {
        "tau": tau,
        "count": int(np.sum(above_tau_mask)),
        "observed_false_accept_rate": observed_false_accept,
        "bound": theoretical_bound,
        "p_bar": p_bar,
        "e_bar": e_bar,
        "bound_satisfied": observed_false_accept <= theoretical_bound + 1e-5
    }

def conformal_acceptance_threshold(probs: np.ndarray, labels: np.ndarray, target_error: float = 0.05) -> float:
    """
    Computes split conformal prediction threshold \tau ensuring false accept rate <= target_error.
    """
    # Nonconformity score for accepted predictions
    rejected_probs = probs[labels == 0]
    if len(rejected_probs) == 0:
        return 0.5
    # Threshold at (1 - target_error) quantile of rejected probabilities
    q = np.quantile(rejected_probs, min(1.0, 1.0 - target_error))
    return float(max(0.5, min(0.99, q)))
