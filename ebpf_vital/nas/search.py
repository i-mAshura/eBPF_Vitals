r"""
Verifier-Constrained Differentiable Neural Architecture Search and Distillation Engine.
Optimizes:
min_\phi L_task(\phi) + \lambda * ReLU(1 - \hat{V}_\theta(\phi)) + \mu * ReLU(C(\phi) - B)
Ref: Eq. (1), (2), Section 6 & Figure 2 of eBPF-VITAL paper.
"""

from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .supernet import DifferentiableSupernet
from .distillation import UserspaceTeacher, TaskLoss
from ..surrogate.gnn_surrogate import VerifierSurrogate
from ..verifier.abstract_interpreter import BPFVerifier, VerificationResult
from ..verifier.programs import ArchitectureSpec, build_neural_bpf_program

class VerifierConstrainedNAS:
    def __init__(self,
                 supernet: DifferentiableSupernet,
                 surrogate: VerifierSurrogate,
                 teacher: UserspaceTeacher,
                 verifier: BPFVerifier,
                 budget_b: float = 25.0, # in 10k instructions (e.g. 250k instructions)
                 lambda_verif: float = 2.0,
                 mu_budget: float = 1.0):
        self.supernet = supernet
        self.surrogate = surrogate
        self.teacher = teacher
        self.verifier = verifier
        self.budget_b = budget_b
        self.lambda_verif = lambda_verif
        self.mu_budget = mu_budget
        self.task_loss_fn = TaskLoss()

    def search_epoch(self,
                     x_batch: torch.Tensor,
                     y_batch: torch.Tensor,
                     stl_batch: Optional[torch.Tensor],
                     optimizer_weights: optim.Optimizer,
                     optimizer_arch: optim.Optimizer,
                     current_lambda: float,
                     current_mu: float) -> Dict[str, float]:
        """
        Executes one alternating or joint step of differentiable search:
        Updates supernet weights and architecture parameters with verifier constraint penalty.
        """
        self.supernet.train()
        self.teacher.eval()
        self.surrogate.eval()

        with torch.no_grad():
            teacher_logits = self.teacher(x_batch)

        # 1. Step Architecture Parameters (alpha)
        optimizer_arch.zero_grad()
        student_logits = self.supernet(x_batch)
        task_loss, metrics = self.task_loss_fn(student_logits, y_batch, teacher_logits, stl_batch)

        # Compute verifier surrogate penalty
        phi = self.supernet.get_arch_vector()
        v_prob, v_budget, _ = self.surrogate.forward_arch(phi)

        verif_penalty = F.relu(1.0 - v_prob.squeeze(-1))
        budget_penalty = F.relu(v_budget.squeeze(-1) - self.budget_b)

        total_arch_loss = task_loss + current_lambda * verif_penalty + current_mu * budget_penalty
        total_arch_loss.backward(retain_graph=True)
        optimizer_arch.step()

        # 2. Step Supernet Weights
        optimizer_weights.zero_grad()
        student_logits_2 = self.supernet(x_batch)
        task_loss_2, _ = self.task_loss_fn(student_logits_2, y_batch, teacher_logits, stl_batch)
        task_loss_2.backward()
        optimizer_weights.step()

        metrics.update({
            "v_prob": float(v_prob.item()),
            "v_budget": float(v_budget.item()),
            "verif_penalty": float(verif_penalty.item()),
            "budget_penalty": float(budget_penalty.item()),
            "total_arch_loss": float(total_arch_loss.item())
        })
        return metrics

    def run_full_search(self,
                        data_loader,
                        num_warmup_epochs: int = 2,
                        num_joint_epochs: int = 5,
                        lr_weights: float = 0.01,
                        lr_arch: float = 0.02) -> Tuple[ArchitectureSpec, VerificationResult]:
        """
        Runs the 3-phase search and ground-truth validation reject loop:
        Phase 1: Warmup
        Phase 2: Joint optimization with increasing lambda, mu
        Phase 3: Discretize and verify on target kernel
        """
        # Separate weight parameters from architecture parameters
        weight_params = [p for n, p in self.supernet.named_parameters() if not n.startswith('alpha')]
        arch_params = [self.supernet.alpha_widths, self.supernet.alpha_depth]

        opt_weights = optim.Adam(weight_params, lr=lr_weights)
        opt_arch = optim.Adam(arch_params, lr=lr_arch)

        # Phase 1: Warmup
        for epoch in range(num_warmup_epochs):
            for x, y, stl in data_loader:
                opt_weights.zero_grad()
                logits = self.supernet(x)
                with torch.no_grad():
                    t_logits = self.teacher(x)
                loss, _ = self.task_loss_fn(logits, y, t_logits, stl)
                loss.backward()
                opt_weights.step()

        # Phase 2: Joint Optimization with scheduled penalties
        for epoch in range(num_joint_epochs):
            prog = float(epoch + 1) / float(num_joint_epochs)
            cur_lambda = self.lambda_verif * prog
            cur_mu = self.mu_budget * prog
            for x, y, stl in data_loader:
                self.search_epoch(x, y, stl, opt_weights, opt_arch, cur_lambda, cur_mu)

        # Phase 3 & 4: Discretization and Ground-Truth Verifier Validation Loop
        discrete_config = self.supernet.discretize()
        spec = ArchitectureSpec(
            input_dim=discrete_config["input_dim"],
            hidden_dims=discrete_config["hidden_dims"],
            output_dim=discrete_config["output_dim"],
            quant_bits=discrete_config["quant_bits"],
            unroll_factor=discrete_config["unroll_factor"],
            use_map_scratch=discrete_config["use_map_scratch"],
            tail_calls=discrete_config["tail_calls"]
        )

        bpf_prog = build_neural_bpf_program(spec)
        verif_res = self.verifier.verify(bpf_prog)

        # Reject loop: if rejected by target verifier, adjust spec until verified
        attempts = 0
        while not verif_res.accepted and attempts < 3:
            attempts += 1
            # Shrink width slightly or increase tail calls
            if spec.hidden_dims:
                spec.hidden_dims = [max(8, int(w * 0.75)) for w in spec.hidden_dims]
            spec.unroll_factor = 1
            spec.use_map_scratch = True
            bpf_prog = build_neural_bpf_program(spec)
            verif_res = self.verifier.verify(bpf_prog)

        return spec, verif_res
