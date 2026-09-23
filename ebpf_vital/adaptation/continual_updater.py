r"""
Projected Online Parameter Update (Algorithm 1) with Bisection Line Search and Tangential Repair.
Ref: Algorithm 1, Section 7 of eBPF-VITAL paper.
"""

from typing import Dict, Tuple, Optional, Callable
import torch
import torch.nn.functional as F
from ..surrogate.gnn_surrogate import VerifierSurrogate

class ContinualUpdater:
    def __init__(self,
                 surrogate: VerifierSurrogate,
                 tau: float = 0.9,
                 epsilon: float = 0.01,
                 lambda_ewc: float = 0.1):
        self.surrogate = surrogate
        self.tau = tau
        self.epsilon = epsilon
        self.lambda_ewc = lambda_ewc
        self.fisher_information: Optional[torch.Tensor] = None

    def update_fisher(self, params: torch.Tensor, grad_sample: torch.Tensor):
        """Updates empirical Fisher information diagonal for EWC."""
        if self.fisher_information is None:
            self.fisher_information = grad_sample.pow(2).detach()
        else:
            self.fisher_information = 0.9 * self.fisher_information + 0.1 * grad_sample.pow(2).detach()

    def compute_ewc_loss(self, current_params: torch.Tensor, committed_params: torch.Tensor) -> torch.Tensor:
        """Elastic Weight Consolidation penalty anchored at last committed parameters."""
        if self.fisher_information is None:
            return torch.tensor(0.0, device=current_params.device)
        ewc = 0.5 * self.lambda_ewc * (self.fisher_information * (current_params - committed_params).pow(2)).sum()
        return ewc

    def project_update(self,
                       phi_t: torch.Tensor,
                       loss_grad: torch.Tensor,
                       lr: float = 0.05,
                       committed_anchor: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        r"""
        Implements Algorithm 1:
        phi_{t+1} = \Pi_{\hat{V}_\theta \ge \tau} ( phi_t - \eta \nabla L_task(phi_t) )
        """
        phi_t = phi_t.clone().detach()

        # Add EWC regularization gradient if anchored
        effective_grad = loss_grad.clone()
        if committed_anchor is not None and self.fisher_information is not None:
            ewc_grad = self.lambda_ewc * self.fisher_information * (phi_t - committed_anchor)
            effective_grad = effective_grad + ewc_grad

        z = phi_t - lr * effective_grad

        # Check line 2: unconstrained step already in S_\tau
        self.surrogate.eval()
        with torch.no_grad():
            prob_z = self.surrogate.predict_acceptance(z)
            p_val = float(prob_z.item())

        if p_val >= self.tau:
            return z, {
                "projected": False,
                "alpha": 1.0,
                "surrogate_prob": p_val,
                "repaired": False
            }

        # Line search bisection: lines 5-14
        alpha_lo = 0.0
        alpha_hi = 1.0

        for _ in range(12): # <= 12 iterations achieves tolerance < 0.0003
            if alpha_hi - alpha_lo <= self.epsilon:
                break
            alpha_mid = (alpha_lo + alpha_hi) / 2.0
            candidate = phi_t + alpha_mid * (z - phi_t)
            with torch.no_grad():
                cand_prob = float(self.surrogate.predict_acceptance(candidate).item())
            if cand_prob >= self.tau:
                alpha_lo = alpha_mid
            else:
                alpha_hi = alpha_mid

        repaired = False
        # Line 15: Tangential repair if alpha_lo is too small (< epsilon)
        if alpha_lo < self.epsilon:
            repaired = True
            # Compute gradient of surrogate acceptance probability w.r.t phi
            phi_req = phi_t.clone().requires_grad_(True)
            prob_req, _, reason_logits = self.surrogate.forward_arch(phi_req)
            prob_req.backward()
            grad_v = phi_req.grad

            if grad_v is not None and grad_v.norm() > 1e-6:
                unit_grad = grad_v / grad_v.norm()
                step_vec = z - phi_t
                # Project step perpendicularly to grad_v (tangential to the decision boundary)
                tangential_step = step_vec - torch.dot(step_vec, unit_grad) * unit_grad
                z_repaired = phi_t + tangential_step

                # Re-run bisection with repaired candidate
                alpha_lo = 0.0
                alpha_hi = 1.0
                for _ in range(10):
                    if alpha_hi - alpha_lo <= self.epsilon:
                        break
                    alpha_mid = (alpha_lo + alpha_hi) / 2.0
                    candidate = phi_t + alpha_mid * (z_repaired - phi_t)
                    with torch.no_grad():
                        cand_prob = float(self.surrogate.predict_acceptance(candidate).item())
                    if cand_prob >= self.tau:
                        alpha_lo = alpha_mid
                    else:
                        alpha_hi = alpha_mid
                final_phi = phi_t + alpha_lo * (z_repaired - phi_t)
            else:
                final_phi = phi_t
        else:
            final_phi = phi_t + alpha_lo * (z - phi_t)

        with torch.no_grad():
            final_prob = float(self.surrogate.predict_acceptance(final_phi).item())

        return final_phi, {
            "projected": True,
            "alpha": float(alpha_lo),
            "surrogate_prob": final_prob,
            "repaired": repaired
        }
