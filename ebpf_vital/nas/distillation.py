r"""
Userspace Teacher Model, Knowledge Distillation, and STL Robustness Loss.
Ref: Eq. (2) and Section 6 of eBPF-VITAL paper.
"""

from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class UserspaceTeacher(nn.Module):
    """
    Unconstrained userspace neurosymbolic teacher model.
    Operates in float32 without eBPF verifier instruction or stack constraints.
    """
    def __init__(self, input_dim: int = 16, hidden_dim: int = 128, num_classes: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 3.0) -> torch.Tensor:
    """
    Kullback-Leibler divergence distillation loss.
    L_KD = T^2 * KL(softmax(z_s / T), softmax(z_t / T))
    """
    p_student = F.log_softmax(student_logits / temperature, dim=-1)
    p_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    kl = F.kl_div(p_student, p_teacher, reduction='batchmean')
    return kl * (temperature ** 2)

def stl_robustness_loss(student_logits: torch.Tensor, stl_robustness: torch.Tensor) -> torch.Tensor:
    """
    Penalizes disagreement between neural confidence and quantitative STL robustness.
    L_STL = Huber(sigmoid(max_logit) - normalized_robustness)
    """
    # Softmax probability of top class
    probs = F.softmax(student_logits, dim=-1)
    top_prob = probs.max(dim=-1)[0]
    # Robustness normalized to [0, 1]
    norm_robustness = torch.sigmoid(stl_robustness)
    return F.smooth_l1_loss(top_prob, norm_robustness)

class TaskLoss(nn.Module):
    """
    Total task loss from Eq. (2):
    L_task = L_CE + \gamma * L_KD + \eta * L_STL
    """
    def __init__(self, gamma: float = 0.5, eta: float = 0.3, kd_temperature: float = 3.0):
        super().__init__()
        self.gamma = gamma
        self.eta = eta
        self.kd_temperature = kd_temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self,
                student_logits: torch.Tensor,
                labels: torch.Tensor,
                teacher_logits: Optional[torch.Tensor] = None,
                stl_robustness: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 1. Supervised Cross-Entropy
        l_ce = self.ce(student_logits, labels)

        # 2. Distillation from Teacher
        if teacher_logits is not None:
            l_kd = distillation_loss(student_logits, teacher_logits, self.kd_temperature)
        else:
            l_kd = torch.tensor(0.0, device=student_logits.device)

        # 3. Temporal Robustness Agreement
        if stl_robustness is not None:
            l_stl = stl_robustness_loss(student_logits, stl_robustness)
        else:
            l_stl = torch.tensor(0.0, device=student_logits.device)

        total = l_ce + self.gamma * l_kd + self.eta * l_stl
        metrics = {
            "ce_loss": float(l_ce.item()),
            "kd_loss": float(l_kd.item()),
            "stl_loss": float(l_stl.item()),
            "task_loss": float(total.item())
        }
        return total, metrics
