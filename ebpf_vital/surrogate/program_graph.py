"""
Program Graph Extraction and Heterogeneous Feature Representation.
Converts BPF instruction streams into control-flow, data-flow, and stack dependence graphs
with abstract-interpretation priors as node features.
Ref: Allamanis et al., "Learning to Represent Programs with Graphs", 2018.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import torch
from ..verifier.abstract_interpreter import BPFInstruction

# Node feature dimensions:
# [0..9]: One-hot instruction category (MOV, ADD/SUB, MUL/DIV, BITWISE, JMP, CALL, EXIT, STX, LDX, OTHER)
# [10]: Register destination index normalized [0..10]
# [11]: Register source index normalized [0..10] or 0
# [12]: Immediate value normalized / log-scaled
# [13]: Stack offset normalized
# [14]: Estimated stack depth at this point
# [15]: Abstract interval range width (smax - smin) normalized
NODE_FEATURE_DIM = 16

OPCODE_CATEGORIES = {
    'MOV': 0,
    'ADD': 1, 'SUB': 1,
    'MUL': 2, 'DIV': 2,
    'AND': 3, 'OR': 3, 'LSH': 3, 'RSH': 3, 'ARSH': 3,
    'JEQ': 4, 'JNE': 4, 'JGT': 4, 'JGE': 4, 'JLT': 4, 'JLE': 4,
    'CALL': 5, 'TAIL_CALL': 5,
    'EXIT': 6,
    'STX': 7,
    'LDX': 8
}

@dataclass
class ProgramGraphData:
    x: torch.Tensor          # [num_nodes, NODE_FEATURE_DIM]
    edge_index: torch.Tensor # [2, num_edges]
    edge_type: torch.Tensor  # [num_edges] (0: CFG, 1: DFG, 2: STACK)
    num_nodes: int

def program_to_graph(instructions: List[BPFInstruction]) -> ProgramGraphData:
    """
    Extracts control-flow and data-flow edges and abstract interpretation prior features.
    """
    num_nodes = max(1, len(instructions))
    node_features = torch.zeros((num_nodes, NODE_FEATURE_DIM), dtype=torch.float32)

    edges_src: List[int] = []
    edges_dst: List[int] = []
    edge_types: List[int] = []

    last_writer: Dict[int, int] = {}
    last_stack_writer: Dict[int, int] = {}
    current_stack_depth = 0

    for i, insn in enumerate(instructions):
        op = insn.op.upper()
        cat = OPCODE_CATEGORIES.get(op, 9)
        node_features[i, cat] = 1.0

        node_features[i, 10] = float(insn.dst) / 10.0
        node_features[i, 11] = float(insn.src) / 10.0 if insn.src is not None else 0.0

        # Immediate scaling
        imm_scaled = float(insn.imm) / 1000.0 if abs(insn.imm) < 10000 else (1.0 if insn.imm > 0 else -1.0)
        node_features[i, 12] = imm_scaled

        # Stack offset
        node_features[i, 13] = float(insn.off) / 512.0

        if op == 'STX' and insn.off < 0:
            current_stack_depth = max(current_stack_depth, abs(insn.off))
        node_features[i, 14] = float(current_stack_depth) / 512.0

        # Register interval estimation prior
        node_features[i, 15] = 0.5 # default range prior

        # 1. Control-Flow Edges (CFG)
        if op == 'EXIT':
            pass
        elif op.startswith('J'):
            # Fallthrough
            if i + 1 < num_nodes:
                edges_src.append(i)
                edges_dst.append(i + 1)
                edge_types.append(0)
            # Branch target
            target = insn.target
            if 0 <= target < num_nodes:
                edges_src.append(i)
                edges_dst.append(target)
                edge_types.append(0)
        else:
            # Sequential CFG edge
            if i + 1 < num_nodes:
                edges_src.append(i)
                edges_dst.append(i + 1)
                edge_types.append(0)

        # 2. Data-Flow Edges (DFG: register read after write)
        if insn.src is not None and insn.src in last_writer:
            src_node = last_writer[insn.src]
            edges_src.append(src_node)
            edges_dst.append(i)
            edge_types.append(1)

        # Update writer
        if op in ('MOV', 'ADD', 'SUB', 'MUL', 'DIV', 'AND', 'OR', 'LSH', 'RSH', 'ARSH', 'LDX'):
            last_writer[insn.dst] = i

        # 3. Stack Dependence Edges
        if op == 'STX':
            last_stack_writer[insn.off] = i
        elif op == 'LDX' and insn.off in last_stack_writer:
            src_node = last_stack_writer[insn.off]
            edges_src.append(src_node)
            edges_dst.append(i)
            edge_types.append(2)

    # Self-loops if no edges exist
    if not edges_src:
        for i in range(num_nodes):
            edges_src.append(i)
            edges_dst.append(i)
            edge_types.append(0)

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_type = torch.tensor(edge_types, dtype=torch.long)

    return ProgramGraphData(
        x=node_features,
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=num_nodes
    )
