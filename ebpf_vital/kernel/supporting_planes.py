r"""
Supporting Kernel Infrastructure Planes:
1. Verdict-Aware LSM Provenance and Taint Capture Plane
2. Graduated Enforcement Plane (Allow -> Rate-limit -> Quarantine -> Kill) with Shadow Mode
3. Self-Protection Meta-Monitor and Heartbeat Watchdog
Ref: Section 8 of eBPF-VITAL paper.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Set, Tuple, Optional
import time

class EnforcementAction(Enum):
    ALLOW = 0
    RATE_LIMIT = 1
    QUARANTINE_CGROUP = 2
    KILL_PROCESS = 3

@dataclass
class ProvenanceNode:
    node_id: int
    obj_type: str         # 'PROCESS', 'FILE', 'SOCKET', 'PIPE'
    cgroup_id: int
    taint_tags: int = 0   # 32-bit taint bitmask

@dataclass
class LSMDecisionEvent:
    hook_name: str        # e.g. 'bprm_check_security', 'file_open', 'socket_connect'
    pid: int
    cgroup_id: int
    src_obj_id: int
    dst_obj_id: int
    lsm_verdict: int      # 0 = ALLOW, -EPERM = DENY
    args_hash: int
    timestamp: float

class LSMProvenancePlane:
    """
    Verdict-aware LSM provenance and taint capture plane.
    Avoids TOCTOU races by attaching to LSM decisions.
    Scoping maps per cgroup provides tenant isolation and bounds state.
    """
    def __init__(self, max_nodes_per_cgroup: int = 10000):
        self.max_nodes = max_nodes_per_cgroup
        # Maps keyed by (cgroup_id, object_id)
        self.nodes: Dict[Tuple[int, int], ProvenanceNode] = {}
        # Ring buffer of recent edges
        self.edges: List[Tuple[int, int, str, float]] = []

    def get_or_create_node(self, cgroup_id: int, obj_id: int, obj_type: str, initial_taint: int = 0) -> ProvenanceNode:
        key = (cgroup_id, obj_id)
        if key not in self.nodes:
            if len(self.nodes) >= self.max_nodes:
                # Evict oldest entry
                oldest_k = next(iter(self.nodes))
                del self.nodes[oldest_k]
            self.nodes[key] = ProvenanceNode(obj_id, obj_type, cgroup_id, initial_taint)
        return self.nodes[key]

    def record_operation(self, event: LSMDecisionEvent) -> Tuple[int, bool]:
        """
        Records operation, combines taints (union), and checks dataflow across cgroups.
        Returns: (new_destination_taint, is_cross_cgroup)
        """
        src = self.get_or_create_node(event.cgroup_id, event.src_obj_id, 'PROCESS')
        dst = self.get_or_create_node(event.cgroup_id, event.dst_obj_id, 'FILE')

        # Propagate taint: destination receives union of source taints
        dst.taint_tags |= src.taint_tags
        self.edges.append((event.src_obj_id, event.dst_obj_id, event.hook_name, event.timestamp))

        # Check cross-cgroup flow
        is_cross = (src.cgroup_id != dst.cgroup_id)
        return dst.taint_tags, is_cross

class GraduatedEnforcementPlane:
    """
    Translates neural gate scores and falsified temporal subformulas into graduated actions.
    Supports Shadow Mode for safe production rollouts.
    """
    def __init__(self, shadow_mode: bool = False):
        self.shadow_mode = shadow_mode
        self.action_history: List[Dict] = []
        self.quarantined_cgroups: Set[int] = set()
        self.killed_pids: Set[int] = set()

    def determine_action(self,
                         anomaly_score: float,
                         is_subformula_falsified: bool,
                         pid: int,
                         cgroup_id: int) -> EnforcementAction:
        """Graduated decision ladder based on confidence and temporal logic falsification."""
        if not is_subformula_falsified and anomaly_score < 0.5:
            action = EnforcementAction.ALLOW
        elif anomaly_score < 0.7 and not is_subformula_falsified:
            action = EnforcementAction.RATE_LIMIT
        elif anomaly_score < 0.9:
            action = EnforcementAction.QUARANTINE_CGROUP
        else:
            action = EnforcementAction.KILL_PROCESS

        # Execute or shadow
        if not self.shadow_mode:
            if action == EnforcementAction.QUARANTINE_CGROUP:
                self.quarantined_cgroups.add(cgroup_id)
            elif action == EnforcementAction.KILL_PROCESS:
                self.killed_pids.add(pid)

        self.action_history.append({
            "pid": pid,
            "cgroup_id": cgroup_id,
            "anomaly_score": anomaly_score,
            "falsified": is_subformula_falsified,
            "action": action,
            "executed": not self.shadow_mode
        })
        return action

class SelfProtectionMetaMonitor:
    """
    Protects eBPF security monitors from tampering:
    1. Mediates bpf() syscalls: only allows signed loader to modify maps/programs
    2. Heartbeat watchdog: detects probe detachment or silencing within 200ms
    """
    def __init__(self, authorized_loader_pid: int = 1001, heartbeat_interval_ms: int = 100):
        self.authorized_pid = authorized_loader_pid
        self.interval_ms = heartbeat_interval_ms
        self.last_heartbeat_time = time.time()
        self.tamper_detected = False
        self.tamper_alerts: List[str] = []

    def record_heartbeat(self):
        """Called by timer-driven BPF program to signal monitor liveness."""
        self.last_heartbeat_time = time.time()

    def check_liveness(self) -> bool:
        """Watchdog check: triggers alarm if heartbeat delayed > 2 * interval."""
        elapsed_ms = (time.time() - self.last_heartbeat_time) * 1000.0
        if elapsed_ms > (2 * self.interval_ms):
            self.tamper_detected = True
            self.tamper_alerts.append(f"Probe detachment detected: heartbeat missed for {elapsed_ms:.1f}ms")
            return False
        return True

    def mediate_bpf_syscall(self, calling_pid: int, cmd: str) -> bool:
        """
        LSM hook mediation on bpf() syscall.
        Blocks unauthorized detach, map write, or program replacement.
        """
        if cmd in ('BPF_PROG_LOAD', 'BPF_MAP_UPDATE_ELEM', 'BPF_PROG_DETACH'):
            if calling_pid != self.authorized_pid:
                self.tamper_detected = True
                self.tamper_alerts.append(f"Unauthorized bpf({cmd}) attempt by pid {calling_pid}")
                return False # Block
        return True # Allow
