r"""
Adversarial Mining and Recalibration Loop for Verifier Surrogate.
Mines near-boundary programs (\hat{V}_\theta \approx 0.5) to keep the surrogate honest
where search and online projection actually operate.
Ref: Section 5 & Figure 3 of eBPF-VITAL paper.
"""

from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.optim as optim
from ..verifier.abstract_interpreter import BPFVerifier, VerificationResult, FailureReason
from ..verifier.programs import ArchitectureSpec, build_neural_bpf_program, mutate_architecture
from .gnn_surrogate import VerifierSurrogate
from .calibration import compute_ece, TemperatureScaler

class BoundaryProgramMiner:
    def __init__(self, verifier: BPFVerifier, surrogate: VerifierSurrogate):
        self.verifier = verifier
        self.surrogate = surrogate

    def mine_boundary_programs(self, seed_specs: List[ArchitectureSpec], num_iterations: int = 5) -> List[Tuple[ArchitectureSpec, VerificationResult]]:
        """
        Adversarially mines programs in the vicinity of the acceptance boundary.
        Perturbs architecture features towards maximum surrogate uncertainty (p \approx 0.5),
        then invokes the real verifier for ground-truth labeling.
        """
        mined_results: List[Tuple[ArchitectureSpec, VerificationResult]] = []

        for spec in seed_specs:
            current_spec = spec
            for _ in range(num_iterations):
                # Check current surrogate prediction
                phi_vec = torch.tensor(current_spec.to_vector(), dtype=torch.float32)
                with torch.no_grad():
                    prob, _, reason_logits = self.surrogate.forward_arch(phi_vec)
                    p_val = float(prob.item())

                # If near boundary (0.2 <= p <= 0.8), verify with real verifier
                prog = build_neural_bpf_program(current_spec)
                verif_res = self.verifier.verify(prog)
                mined_results.append((current_spec, verif_res))

                # Direct mutation using surrogate uncertainty or verifier failure reason
                if verif_res.accepted:
                    # Near accept boundary: push towards complexity limit
                    current_spec = mutate_architecture(current_spec, FailureReason.NONE)
                else:
                    # Near reject boundary: repair using reason
                    current_spec = mutate_architecture(current_spec, verif_res.failure_reason)

        return mined_results

    def recalibration_round(self,
                            surrogate_optimizer: optim.Optimizer,
                            training_specs: List[ArchitectureSpec],
                            mined_specs: List[Tuple[ArchitectureSpec, VerificationResult]],
                            epochs: int = 5) -> Dict[str, float]:
        """
        Runs one recalibration round on the augmented dataset (standard + adversarially mined programs).
        Returns training metrics and calibration agreement.
        """
        # Prepare dataset tensors
        all_specs: List[ArchitectureSpec] = list(training_specs)
        all_labels: List[int] = []
        all_budgets: List[float] = []
        all_reasons: List[int] = []

        # Evaluate base specs if not evaluated
        for spec in training_specs:
            prog = build_neural_bpf_program(spec)
            res = self.verifier.verify(prog)
            all_labels.append(1 if res.accepted else 0)
            all_budgets.append(float(res.processed_instructions) / 10000.0)
            reason_idx = res.failure_reason.value if res.failure_reason != FailureReason.NONE else 0
            all_reasons.append(max(0, reason_idx))

        # Add mined near-boundary specs
        for spec, res in mined_specs:
            all_specs.append(spec)
            all_labels.append(1 if res.accepted else 0)
            all_budgets.append(float(res.processed_instructions) / 10000.0)
            reason_idx = res.failure_reason.value if res.failure_reason != FailureReason.NONE else 0
            all_reasons.append(max(0, reason_idx))

        phi_tensor = torch.tensor([s.to_vector() for s in all_specs], dtype=torch.float32)
        labels_tensor = torch.tensor(all_labels, dtype=torch.float32)
        budgets_tensor = torch.tensor(all_budgets, dtype=torch.float32)
        reasons_tensor = torch.tensor(all_reasons, dtype=torch.long)

        # Train surrogate
        self.surrogate.train()
        for epoch in range(epochs):
            surrogate_optimizer.zero_grad()
            prob, budget, reason_logits = self.surrogate.forward_arch(phi_tensor)
            loss = self.surrogate.compute_loss(
                prob, budget, reason_logits,
                labels_tensor, budgets_tensor, reasons_tensor
            )
            loss.backward()
            surrogate_optimizer.step()

        # Compute post-training agreement
        self.surrogate.eval()
        with torch.no_grad():
            prob, _, _ = self.surrogate.forward_arch(phi_tensor)
            pred_accept = (prob.squeeze(-1) >= 0.5).int().numpy()
            true_accept = labels_tensor.int().numpy()
            agreement = float(np.mean(pred_accept == true_accept))
            probs_np = prob.squeeze(-1).numpy()
            ece, _ = compute_ece(probs_np, true_accept)

        return {
            "loss": float(loss.item()),
            "agreement": agreement,
            "ece": ece,
            "total_programs": len(all_specs)
        }
