import pytest
import torch
from ebpf_vital.surrogate.gnn_surrogate import VerifierSurrogate
from ebpf_vital.verifier.abstract_interpreter import BPFVerifier
from ebpf_vital.verifier.programs import ArchitectureSpec
from ebpf_vital.adaptation.continual_updater import ContinualUpdater
from ebpf_vital.adaptation.checkpoint_protocol import TwoTierCheckpointProtocol

def test_algorithm1_bisection_projection():
    surr = VerifierSurrogate()
    updater = ContinualUpdater(surr, tau=0.8, epsilon=0.01)

    phi_t = torch.tensor([16.0, 2.0, 32.0, 16.0, 0.0, 5.0, 8.0, 2.0, 1.0, 1.0, 1.0])
    loss_grad = torch.randn(11) * 0.5

    phi_next, info = updater.project_update(phi_t, loss_grad, lr=0.1)
    assert phi_next.shape == (11,)
    assert "projected" in info
    assert "surrogate_prob" in info

def test_algorithm2_two_tier_checkpoint():
    surr = VerifierSurrogate()
    verifier = BPFVerifier(budget=50000)
    spec = ArchitectureSpec(input_dim=16, hidden_dims=[16, 8], output_dim=2)

    updater = ContinualUpdater(surr, tau=0.8, epsilon=0.01)
    protocol = TwoTierCheckpointProtocol(updater, verifier, spec, checkpoint_interval_k=3)

    for i in range(6):
        grad = torch.randn(11) * 0.1
        step_res = protocol.step(grad, lr=0.02)
        assert "step" in step_res
        assert step_res["step"] == i + 1

    # Invariant: committed state exists and must pass verifier
    committed_spec = ArchitectureSpec(
        hidden_dims=[int(max(8, protocol.phi_c[2].item())), int(max(8, protocol.phi_c[3].item()))]
    )
    from ebpf_vital.verifier.programs import build_neural_bpf_program
    committed_prog = build_neural_bpf_program(committed_spec)
    verif_res = verifier.verify(committed_prog)
    assert verif_res.accepted
