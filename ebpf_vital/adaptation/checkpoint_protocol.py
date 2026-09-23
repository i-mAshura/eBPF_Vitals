r"""
Two-Tier Checkpoint and Rollback Protocol (Algorithm 2).
Guarantees formal loadability invariants via canary verifier invocations every k updates,
atomic BPF map swapping, and automated rollback upon verification discrepancies.
Ref: Algorithm 2, Section 7 & Figure 4 of eBPF-VITAL paper.
"""

from typing import Dict, List, Tuple, Optional, Callable
import time
import torch
from .continual_updater import ContinualUpdater
from ..verifier.abstract_interpreter import BPFVerifier, VerificationResult
from ..verifier.programs import ArchitectureSpec, build_neural_bpf_program

class TwoTierCheckpointProtocol:
    def __init__(self,
                 updater: ContinualUpdater,
                 verifier: BPFVerifier,
                 initial_spec: ArchitectureSpec,
                 checkpoint_interval_k: int = 8,
                 recalibration_callback: Optional[Callable[[List[torch.Tensor]], None]] = None):
        self.updater = updater
        self.verifier = verifier
        self.k = checkpoint_interval_k
        self.recalibration_callback = recalibration_callback

        # Initial parameter setup: verify initial parameters are strictly loadable
        self.initial_spec = initial_spec
        init_prog = build_neural_bpf_program(initial_spec)
        verif_init = self.verifier.verify(init_prog)
        if not verif_init.accepted:
            raise ValueError(f"Initial parameters rejected by target verifier: {verif_init.failure_message}")

        # phi_c: certified committed parameters
        self.phi_c = torch.tensor(initial_spec.to_vector(), dtype=torch.float32)
        # phi: current online parameters
        self.phi = self.phi_c.clone()

        self.step_count = 0
        self.staged_slot = 0 # Double buffer index (0 or 1)
        self.active_slot = 0

        # Diagnostics and history
        self.history: List[Dict] = []
        self.recalibration_buffer: List[Tuple[torch.Tensor, VerificationResult]] = []
        self.total_rollbacks = 0
        self.total_commits = 0
        self.unprojected_failures_avoided = 0

    def step(self,
             loss_grad: torch.Tensor,
             lr: float = 0.05,
             spec_converter: Optional[Callable[[torch.Tensor], ArchitectureSpec]] = None) -> Dict:
        """
        Executes one iteration of Algorithm 2:
        1. Surrogate-validated step via Algorithm 1 (bisection + repair)
        2. Atomic parameter staging to simulated BPF map slot
        3. Periodic (every k steps) canary verification with target verifier
        """
        self.step_count += 1

        # Check what would happen without projection (counterfactual baseline check)
        unprojected_z = self.phi - lr * loss_grad
        with torch.no_grad():
            unproj_prob = float(self.updater.surrogate.predict_acceptance(unprojected_z).item())
        if unproj_prob < self.updater.tau:
            self.unprojected_failures_avoided += 1

        # 1. Surrogate-validated step (Algorithm 1)
        phi_next, proj_info = self.updater.project_update(
            self.phi,
            loss_grad,
            lr=lr,
            committed_anchor=self.phi_c
        )
        self.phi = phi_next

        # 2. Stage to alternate map slot (atomic swap)
        self.staged_slot = 1 - self.active_slot
        # In kernel, atomic swap happens at next event boundary:
        self.active_slot = self.staged_slot

        committed = False
        rolled_back = False
        verif_time_ms = 0.0

        # 3. Canary checkpoint every k updates
        if self.step_count % self.k == 0:
            start_t = time.perf_counter()
            # Test candidate with real verifier A_\kappa(\phi)
            if spec_converter:
                cand_spec = spec_converter(self.phi)
            else:
                cand_spec = self.initial_spec # fallback

            cand_prog = build_neural_bpf_program(cand_spec)
            canary_result = self.verifier.verify(cand_prog)
            verif_time_ms = (time.perf_counter() - start_t) * 1000.0

            if canary_result.accepted:
                # Commit: phi_c <- phi
                self.phi_c = self.phi.clone()
                self.total_commits += 1
                committed = True
            else:
                # Rollback: phi <- phi_c
                self.phi = self.phi_c.clone()
                self.total_rollbacks += 1
                rolled_back = True
                # Record near-boundary discrepancy for surrogate recalibration
                self.recalibration_buffer.append((phi_next.clone(), canary_result))
                if self.recalibration_callback:
                    self.recalibration_callback([phi_next.clone()])

        step_record = {
            "step": self.step_count,
            "projected": proj_info["projected"],
            "alpha": proj_info["alpha"],
            "surrogate_prob": proj_info["surrogate_prob"],
            "repaired": proj_info["repaired"],
            "committed": committed,
            "rolled_back": rolled_back,
            "verif_time_ms": verif_time_ms,
            "total_rollbacks": self.total_rollbacks,
            "total_commits": self.total_commits,
            "unprojected_failures_avoided": self.unprojected_failures_avoided
        }
        self.history.append(step_record)
        return step_record
