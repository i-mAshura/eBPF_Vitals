r"""
Message-Passing Neural Network (MPNN) Verifier Surrogate \hat{V}_\theta.
Predicts:
1. Probability of verifier acceptance p \in [0, 1]
2. Residual verifier complexity budget \hat{c} \ge 0
3. Coarse-grained failure reason \hat{r} \in \mathbb{R}^5
Ref: Eq. (3), Section 5 of eBPF-VITAL paper.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .program_graph import ProgramGraphData, NODE_FEATURE_DIM

class MessagePassingLayer(nn.Module):
    """Relation-aware message passing convolution for CFG, DFG, and Stack dependence edges."""
    def __init__(self, hidden_dim: int, num_relations: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.relation_weights = nn.Parameter(torch.Tensor(num_relations, hidden_dim, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.update_gate = nn.GRUCell(hidden_dim, hidden_dim)
        nn.init.xavier_uniform_(self.relation_weights)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        messages = torch.zeros_like(x)

        if edge_index.size(1) > 0:
            src, dst = edge_index[0], edge_index[1]
            for rel in range(self.num_relations):
                mask = (edge_type == rel)
                if mask.any():
                    rel_src = src[mask]
                    rel_dst = dst[mask]
                    W = self.relation_weights[rel]
                    rel_msg = torch.matmul(x[rel_src], W)
                    messages.index_add_(0, rel_dst, rel_msg)

        # Update node states using GRU
        updated_x = self.update_gate(messages + self.bias, x)
        return updated_x

class VerifierSurrogate(nn.Module):
    def __init__(self, node_dim: int = NODE_FEATURE_DIM, hidden_dim: int = 64, num_layers: int = 3, arch_dim: int = 11):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.temperature = 1.0  # Learned / calibrated temperature parameter

        # 1. Graph representation branch
        self.node_embed = nn.Linear(node_dim, hidden_dim)
        self.gnn_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim) for _ in range(num_layers)
        ])
        self.graph_readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )

        # 2. Continuous architecture encoding branch (for differentiable NAS and projection)
        self.arch_embed = nn.Sequential(
            nn.Linear(arch_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Combined representation
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 3 Multi-task Readout Heads:
        # Head 1: Verifier Acceptance Logit -> Sigmoid
        self.acceptance_head = nn.Linear(hidden_dim, 1)

        # Head 2: Residual Budget (processed instruction count prediction) -> Softplus
        self.budget_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )

        # Head 3: Reason Head (5 classes: Stack, OOB, Scalar Range, Loop Bound, Complexity)
        self.reason_head = nn.Linear(hidden_dim, 5)

    def forward_graph(self, graph: ProgramGraphData) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass taking a ProgramGraphData object."""
        h = F.relu(self.node_embed(graph.x))
        for layer in self.gnn_layers:
            h = layer(h, graph.edge_index, graph.edge_type)

        # Mean and Max pooling
        h_mean = h.mean(dim=0, keepdim=True)
        h_max = h.max(dim=0, keepdim=True)[0]
        h_pool = torch.cat([h_mean, h_max], dim=-1)
        feat = self.graph_readout(h_pool)

        return self._compute_heads(feat)

    def forward_arch(self, phi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Differentiable forward pass taking continuous architecture vector phi \in [B, arch_dim].
        Used by verifier-constrained NAS and projected continual learning (Algorithms 1 & 2).
        """
        if phi.dim() == 1:
            phi = phi.unsqueeze(0)
        feat = self.arch_embed(phi)
        feat = self.fusion(feat)
        return self._compute_heads(feat)

    def _compute_heads(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Head 1: Acceptance probability with temperature scaling
        logit = self.acceptance_head(feat)
        prob = torch.sigmoid(logit / self.temperature)

        # Head 2: Residual budget prediction
        budget = self.budget_head(feat)

        # Head 3: Reason logits
        reason_logits = self.reason_head(feat)

        return prob, budget, reason_logits

    def predict_acceptance(self, phi: torch.Tensor) -> torch.Tensor:
        """Helper returning acceptance probability \hat{V}_\theta(\phi)."""
        prob, _, _ = self.forward_arch(phi)
        return prob.squeeze(-1)

    def compute_loss(self,
                     pred_prob: torch.Tensor,
                     pred_budget: torch.Tensor,
                     pred_reasons: torch.Tensor,
                     target_accepted: torch.Tensor,
                     target_budget: torch.Tensor,
                     target_reasons: torch.Tensor,
                     alpha: float = 0.5,
                     beta: float = 0.5) -> torch.Tensor:
        """
        Loss formulation from Eq. (3):
        L_sur = BCE(p, A) + \alpha * Huber(c_hat, c) * A + \beta * CE(r_hat, r) * (1 - A)
        """
        # BCE on acceptance
        bce_loss = F.binary_cross_entropy(pred_prob.squeeze(-1), target_accepted.float())

        # Huber loss on residual budget only for accepted programs (A = 1)
        accepted_mask = (target_accepted > 0.5).squeeze(-1) if target_accepted.dim() > 1 else (target_accepted > 0.5)
        if accepted_mask.any():
            huber_loss = F.smooth_l1_loss(
                pred_budget[accepted_mask].squeeze(-1),
                target_budget[accepted_mask].float()
            )
        else:
            huber_loss = torch.tensor(0.0, device=pred_prob.device)

        # Cross Entropy on failure reason only for rejected programs (A = 0)
        rejected_mask = ~accepted_mask
        if rejected_mask.any():
            ce_loss = F.cross_entropy(
                pred_reasons[rejected_mask],
                target_reasons[rejected_mask].long()
            )
        else:
            ce_loss = torch.tensor(0.0, device=pred_prob.device)

        total_loss = bce_loss + alpha * huber_loss + beta * ce_loss
        return total_loss
