import pytest
import numpy as np
import torch
from ebpf_vital.verifier.programs import ArchitectureSpec, build_neural_bpf_program
from ebpf_vital.surrogate.program_graph import program_to_graph
from ebpf_vital.surrogate.gnn_surrogate import VerifierSurrogate
from ebpf_vital.surrogate.calibration import compute_ece, TemperatureScaler, verify_false_accept_bound

def test_program_graph_extraction():
    spec = ArchitectureSpec(input_dim=8, hidden_dims=[16], output_dim=2)
    prog = build_neural_bpf_program(spec)
    graph = program_to_graph(prog)
    assert graph.num_nodes == len(prog)
    assert graph.x.shape == (len(prog), 16)
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_index.shape[1] > 0

def test_surrogate_heads_and_loss():
    surr = VerifierSurrogate()
    phi = torch.randn(4, 11)
    p, c, r = surr.forward_arch(phi)
    assert p.shape == (4, 1)
    assert c.shape == (4, 1)
    assert r.shape == (4, 5)
    # Probabilities in [0, 1]
    assert (p >= 0.0).all() and (p <= 1.0).all()
    # Residual budget non-negative
    assert (c >= 0.0).all()

    # Loss computation
    target_a = torch.tensor([1, 0, 1, 0])
    target_c = torch.tensor([12.0, 0.0, 5.0, 0.0])
    target_r = torch.tensor([0, 1, 0, 4])
    loss = surr.compute_loss(p, c, r, target_a, target_c, target_r)
    assert loss.item() > 0

def test_calibration_and_ece():
    probs = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.95])
    labels = np.array([1, 1, 0, 0, 0, 1])
    ece, stats = compute_ece(probs, labels, num_bins=5)
    assert 0.0 <= ece <= 1.0
    assert len(stats) == 5

    bound_res = verify_false_accept_bound(probs, labels, tau=0.85, ece=ece)
    assert "bound_satisfied" in bound_res
    assert bound_res["count"] > 0
