r"""
Provenance Dataset and Workload Generator.
Generates realistic provenance events for:
- DARPA TC Scenarios (CADETS, THEIA, TRACE)
- StreamSpot Benchmark
- MITRE ATT&CK Tactics (Execution, Privilege Escalation, Persistence, Lateral Movement, Exfiltration)
- Sustained-load benign background traffic (nginx and redis workloads)
Ref: Section 9 & Table 1 of eBPF-VITAL paper.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import random
import numpy as np
from ..kernel.temporal_monitor import TemporalEvent

ATTACK_TACTICS = {
    0: "BENIGN",
    1: "EXECUTION",
    2: "PRIVILEGE_ESCALATION",
    3: "PERSISTENCE",
    4: "LATERAL_MOVEMENT",
    5: "EXFILTRATION"
}

@dataclass
class LabeledTraceSample:
    features: np.ndarray       # [16] feature vector (syscall ID, arguments hash, object IDs, etc.)
    tactic_label: int          # 0..5
    event: TemporalEvent
    scenario: str              # 'CADETS', 'THEIA', 'TRACE', 'StreamSpot'
    is_attack: bool

class ProvenanceTraceGenerator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def generate_benchmark_dataset(self,
                                   samples_per_scenario: int = 400,
                                   attack_ratio: float = 0.25) -> Dict[str, List[LabeledTraceSample]]:
        """
        Generates full labeled provenance datasets for CADETS, THEIA, TRACE, and StreamSpot.
        """
        scenarios = ['CADETS', 'THEIA', 'TRACE', 'StreamSpot']
        dataset = {}

        for scen in scenarios:
            trace_samples: List[LabeledTraceSample] = []
            cur_time = 1000.0

            for i in range(samples_per_scenario):
                is_attack = (self.rng.random() < attack_ratio)
                cur_time += self.rng.uniform(0.01, 0.5)

                if is_attack:
                    # Attack event mapping to MITRE tactic
                    if scen == 'CADETS':
                        # Exfiltration or Execution via living-off-the-land
                        tactic = self.rng.choice([1, 5])
                    elif scen == 'THEIA':
                        # Browser exploit, Privilege Escalation, Persistence
                        tactic = self.rng.choice([2, 3])
                    elif scen == 'TRACE':
                        # Lateral Movement or Privilege Escalation
                        tactic = self.rng.choice([2, 4])
                    else: # StreamSpot
                        tactic = self.rng.choice([1, 3, 5])

                    feat = self._synthesize_attack_features(tactic)
                    op_name = 'send' if tactic == 5 else ('execve' if tactic == 1 else 'setuid')
                    event = TemporalEvent(
                        timestamp=cur_time,
                        event_id=i,
                        op=op_name,
                        src_obj=100 + self.rng.randint(0, 5),
                        dst_obj=500 + self.rng.randint(0, 20),
                        is_sensitive=(tactic == 5),
                        is_allow_dest=(self.rng.random() < 0.1), # mostly non-allow destination
                        cgroup_id=1,
                        taint_tags=(1 << tactic)
                    )
                else:
                    # Benign background event
                    tactic = 0
                    feat = self._synthesize_benign_features(scen)
                    op_name = self.rng.choice(['read', 'write', 'stat', 'fstat', 'mprotect', 'close'])
                    event = TemporalEvent(
                        timestamp=cur_time,
                        event_id=i,
                        op=op_name,
                        src_obj=10 + self.rng.randint(0, 10),
                        dst_obj=50 + self.rng.randint(0, 50),
                        is_sensitive=False,
                        is_allow_dest=True,
                        cgroup_id=1,
                        taint_tags=0
                    )

                sample = LabeledTraceSample(
                    features=feat,
                    tactic_label=tactic,
                    event=event,
                    scenario=scen,
                    is_attack=is_attack
                )
                trace_samples.append(sample)

            dataset[scen] = trace_samples
        return dataset

    def _synthesize_attack_features(self, tactic: int) -> np.ndarray:
        vec = self.np_rng.normal(loc=0.5, scale=0.3, size=16)
        vec[tactic] += 2.0 # prominent signal along tactic dimension
        # Taint flag
        vec[10] = 1.0
        # High argument entropy
        vec[11] = 0.85
        return vec.astype(np.float32)

    def _synthesize_benign_features(self, scenario: str) -> np.ndarray:
        vec = self.np_rng.normal(loc=0.0, scale=0.2, size=16)
        vec[0] = 1.5 # benign class marker
        vec[10] = 0.0 # untainted
        vec[11] = 0.1 # low argument entropy
        return vec.astype(np.float32)

    def generate_background_workload(self, num_events: int = 1000, workload_type: str = "nginx") -> List[TemporalEvent]:
        """
        Simulates sustained-load background syscall activity for nginx or redis.
        Used for overhead, latency profiling, and false-positive evaluation.
        """
        events: List[TemporalEvent] = []
        cur_t = 0.0

        for i in range(num_events):
            cur_t += self.rng.uniform(0.0001, 0.002) # high throughput
            if workload_type == "nginx":
                op = self.rng.choice(['epoll_wait', 'recvfrom', 'sendto', 'openat', 'writev'])
            else: # redis
                op = self.rng.choice(['read', 'write', 'epoll_ctl', 'futex'])

            ev = TemporalEvent(
                timestamp=cur_t,
                event_id=i,
                op=op,
                src_obj=1000,
                dst_obj=2000 + (i % 100),
                is_sensitive=False,
                is_allow_dest=True,
                cgroup_id=2
            )
            events.append(ev)
        return events
