import pytest
import torch
from ebpf_vital.nas.supernet import DifferentiableSupernet, quantize_int8
from ebpf_vital.nas.distillation import UserspaceTeacher, TaskLoss, distillation_loss

def test_int8_quantization_ste():
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
    q = quantize_int8(x)
    loss = (q ** 2).sum()
    loss.backward()
    assert x.grad is not None
    # Straight through estimator delivers gradient
    assert (x.grad != 0).any()

def test_supernet_forward_and_discretization():
    supernet = DifferentiableSupernet(input_dim=16, num_classes=5, candidate_widths=[16, 32], max_layers=2)
    x = torch.randn(8, 16)
    out = supernet(x)
    assert out.shape == (8, 5)

    phi = supernet.get_arch_vector()
    assert phi.shape == (11,)

    discrete = supernet.discretize()
    assert "hidden_dims" in discrete
    assert len(discrete["hidden_dims"]) >= 1

def test_distillation_and_task_loss():
    teacher = UserspaceTeacher(input_dim=16, hidden_dim=32, num_classes=5)
    x = torch.randn(4, 16)
    y = torch.tensor([0, 1, 2, 3])
    t_logits = teacher(x)

    s_logits = torch.randn(4, 5, requires_grad=True)
    loss_fn = TaskLoss()
    total_loss, metrics = loss_fn(s_logits, y, teacher_logits=t_logits)
    assert total_loss.item() > 0
    assert "ce_loss" in metrics
    assert "kd_loss" in metrics
