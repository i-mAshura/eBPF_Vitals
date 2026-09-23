r"""
In-Kernel Bounded Past-Time Metric Temporal Logic (pt-LTL / MTL) Monitor.
Compiles formulas into subformula DAGs in topological order.
Implements Algorithm 3: Budget-Scheduled Monitor Evaluation under per-event cycle budget b.
Publishes 2-bit subformula status (pending, satisfied, falsified) and robustness into shared BPF map slots.
Ref: Section 8 & Algorithm 3 of eBPF-VITAL paper.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Tuple, Optional, Callable
import time

class SubformulaStatus(Enum):
    PENDING = 0
    SATISFIED = 1
    FALSIFIED = 2
    UNKNOWN = 3

@dataclass
class TemporalEvent:
    timestamp: float         # Wall-clock timestamp (seconds)
    event_id: int            # Monotonic event sequence counter
    op: str                  # Syscall / LSM hook operation name (e.g. 'read', 'write', 'send', 'exec')
    src_obj: int             # Source object / process ID
    dst_obj: int             # Destination object / socket / file ID
    is_sensitive: bool = False
    is_allow_dest: bool = False
    cgroup_id: int = 1
    taint_tags: int = 0      # 32-bit bitmask of taint sources

@dataclass
class SubformulaNode:
    node_id: int
    name: str
    op_type: str             # 'ATOM', 'NOT', 'AND', 'OR', 'ONCE', 'HISTORICALLY', 'SINCE'
    children: List[int] = field(default_factory=list)
    time_window: float = 0.0 # Time window horizon in seconds (for ONCE / HISTORICALLY)
    worst_case_cost: int = 10 # Estimated cycle cost
    eval_fn: Optional[Callable[[TemporalEvent, Dict[int, SubformulaStatus]], SubformulaStatus]] = None

    # State per node (in kernel, stored in fixed array map slot)
    current_status: SubformulaStatus = SubformulaStatus.PENDING
    robustness: float = 0.0
    last_satisfied_time: float = -1.0
    history_satisfied: bool = True

class TemporalPropertyDAG:
    """
    Topologically ordered subformula DAG for a past-time specification.
    Linear evaluation time per event; constant memory footprint.
    """
    def __init__(self, property_id: int, name: str, weight: float = 1.0, horizon: float = 30.0, cost: int = 50):
        self.property_id = property_id
        self.name = name
        self.weight = weight
        self.horizon = horizon
        self.worst_case_cost = cost
        self.nodes: List[SubformulaNode] = []
        self.active_since: float = 0.0
        self.is_valid: bool = True

    def add_node(self, node: SubformulaNode):
        self.nodes.append(node)

    def evaluate_event(self, event: TemporalEvent) -> Tuple[SubformulaStatus, float]:
        """Evaluates all subformula nodes in topological order."""
        node_statuses: Dict[int, SubformulaStatus] = {}

        for node in self.nodes:
            if node.op_type == 'ATOM':
                if node.eval_fn:
                    st = node.eval_fn(event, node_statuses)
                else:
                    st = SubformulaStatus.PENDING
                node.current_status = st
                if st == SubformulaStatus.SATISFIED:
                    node.last_satisfied_time = event.timestamp
                    node.robustness = 1.0
                elif st == SubformulaStatus.FALSIFIED:
                    node.robustness = -1.0
                else:
                    node.robustness = 0.0

            elif node.op_type == 'NOT':
                child_st = node_statuses[node.children[0]]
                if child_st == SubformulaStatus.SATISFIED:
                    node.current_status = SubformulaStatus.FALSIFIED
                    node.robustness = -node.nodes_robustness if hasattr(node, 'nodes_robustness') else -1.0
                elif child_st == SubformulaStatus.FALSIFIED:
                    node.current_status = SubformulaStatus.SATISFIED
                    node.robustness = 1.0
                else:
                    node.current_status = child_st
                    node.robustness = 0.0

            elif node.op_type == 'AND':
                st0 = node_statuses[node.children[0]]
                st1 = node_statuses[node.children[1]]
                if st0 == SubformulaStatus.SATISFIED and st1 == SubformulaStatus.SATISFIED:
                    node.current_status = SubformulaStatus.SATISFIED
                    node.robustness = 1.0
                elif st0 == SubformulaStatus.FALSIFIED or st1 == SubformulaStatus.FALSIFIED:
                    node.current_status = SubformulaStatus.FALSIFIED
                    node.robustness = -1.0
                else:
                    node.current_status = SubformulaStatus.PENDING
                    node.robustness = 0.0

            elif node.op_type == 'ONCE': # O_[0, T] child
                child_st = node_statuses[node.children[0]]
                if child_st == SubformulaStatus.SATISFIED:
                    node.last_satisfied_time = event.timestamp

                # Check if child was satisfied within horizon [event.timestamp - T, event.timestamp]
                if node.last_satisfied_time >= 0 and (event.timestamp - node.last_satisfied_time) <= node.time_window:
                    node.current_status = SubformulaStatus.SATISFIED
                    # Robustness decreases as deadline approaches
                    remaining_fraction = 1.0 - (event.timestamp - node.last_satisfied_time) / max(0.001, node.time_window)
                    node.robustness = remaining_fraction
                else:
                    node.current_status = SubformulaStatus.FALSIFIED
                    node.robustness = -1.0

            node_statuses[node.node_id] = node.current_status

        # Root verdict is the last node's status
        root = self.nodes[-1]
        return root.current_status, root.robustness

class BudgetScheduledMonitor:
    """
    Implements Algorithm 3: Budget-Scheduled Monitor Evaluation (per event).
    Maintains active prefixes under cycle budget b, marking deprioritized properties as UNKNOWN.
    """
    def __init__(self, per_event_budget: int = 150):
        self.budget = per_event_budget
        self.properties: List[TemporalPropertyDAG] = []

    def add_property(self, prop: TemporalPropertyDAG):
        self.properties.append(prop)
        # Keep properties sorted by priority (weight / cost ratio)
        self.properties.sort(key=lambda p: p.weight / max(1, p.worst_case_cost), reverse=True)

    def evaluate(self, event: TemporalEvent) -> Dict[str, any]:
        """
        Executes Algorithm 3 per event:
        Lines 1-13 of Algorithm 3.
        """
        spent = 0
        verdicts: Dict[int, SubformulaStatus] = {}
        robustness_map: Dict[int, float] = {}
        partial_state_vector: List[int] = [] # 2-bit status per subformula published to shared map

        for prop in self.properties:
            if spent + prop.worst_case_cost <= self.budget:
                verdict, rob = prop.evaluate_event(event)
                spent += prop.worst_case_cost

                # Reinstated after full horizon of continuous activity
                if not prop.is_valid and (event.timestamp - prop.active_since) >= prop.horizon:
                    prop.is_valid = True

                verdicts[prop.property_id] = verdict
                robustness_map[prop.property_id] = rob
            else:
                # Budget exceeded: mark unknown, do not update state
                prop.is_valid = False
                prop.active_since = event.timestamp
                verdicts[prop.property_id] = SubformulaStatus.UNKNOWN
                robustness_map[prop.property_id] = 0.0

            # Collect partial evaluation states for neural gate
            for node in prop.nodes:
                partial_state_vector.append(node.current_status.value)

        return {
            "spent_cycles": spent,
            "budget": self.budget,
            "verdicts": verdicts,
            "robustness": robustness_map,
            "partial_state_vector": partial_state_vector
        }

def create_apt_exfiltration_dag() -> TemporalPropertyDAG:
    r"""
    Builds the exfiltration property from Eq. (1) of paper:
    \psi_exfil = send \wedge O_[0, 30s] read_sensitive \wedge \neg allow_dest
    """
    dag = TemporalPropertyDAG(property_id=1, name="exfiltration_pattern", weight=2.0, horizon=30.0, cost=40)

    # Node 0: read_sensitive atom
    n0 = SubformulaNode(0, "read_sensitive", "ATOM", eval_fn=lambda e, s: (
        SubformulaStatus.SATISFIED if e.op == 'read' and e.is_sensitive else SubformulaStatus.FALSIFIED
    ))
    # Node 1: O_[0, 30s] (read_sensitive)
    n1 = SubformulaNode(1, "O_30s_read_sensitive", "ONCE", children=[0], time_window=30.0)

    # Node 2: send atom
    n2 = SubformulaNode(2, "is_send", "ATOM", eval_fn=lambda e, s: (
        SubformulaStatus.SATISFIED if e.op in ('send', 'socket_send', 'connect') else SubformulaStatus.FALSIFIED
    ))

    # Node 3: allow_dest atom
    n3 = SubformulaNode(3, "allow_dest", "ATOM", eval_fn=lambda e, s: (
        SubformulaStatus.SATISFIED if e.is_allow_dest else SubformulaStatus.FALSIFIED
    ))

    # Node 4: not_allow_dest = NOT(allow_dest)
    n4 = SubformulaNode(4, "not_allow_dest", "NOT", children=[3])

    # Node 5: send_and_once = AND(is_send, O_30s_read_sensitive)
    n5 = SubformulaNode(5, "send_and_once", "AND", children=[2, 1])

    # Node 6: root = AND(send_and_once, not_allow_dest)
    n6 = SubformulaNode(6, "psi_exfil", "AND", children=[5, 4])

    for n in (n0, n1, n2, n3, n4, n5, n6):
        dag.add_node(n)
    return dag
