import pytest
import numpy as np
from ebpf_vital.verifier.programs import ArchitectureSpec
from ebpf_vital.bpf_codegen.generator import generate_bpf_c_source
from ebpf_vital.kernel.temporal_monitor import BudgetScheduledMonitor, create_apt_exfiltration_dag, TemporalEvent
from ebpf_vital.kernel.neural_gate import InKernelNeuralGate
from ebpf_vital.kernel.supporting_planes import LSMProvenancePlane, LSMDecisionEvent, GraduatedEnforcementPlane, SelfProtectionMetaMonitor, EnforcementAction
from ebpf_vital.evaluation.provenance_generator import ProvenanceTraceGenerator
from ebpf_vital.evaluation.benchmarks import EBPFVitalDetector, evaluate_detector_on_scenarios

def test_full_dataplane_pipeline():
    # 1. Provenance plane
    prov_plane = LSMProvenancePlane()
    lsm_ev = LSMDecisionEvent('bprm_check_security', 101, 1, 10, 20, 0, 1234, 1.0)
    taint, is_cross = prov_plane.record_operation(lsm_ev)
    assert not is_cross

    # 2. Temporal monitor
    monitor = BudgetScheduledMonitor()
    monitor.add_property(create_apt_exfiltration_dag())
    ev1 = TemporalEvent(1.0, 1, 'read', 10, 20, is_sensitive=True)
    m_res1 = monitor.evaluate(ev1)
    ev2 = TemporalEvent(2.0, 2, 'send', 10, 30, is_allow_dest=False)
    m_res2 = monitor.evaluate(ev2)

    # 3. Neural gate
    gate = InKernelNeuralGate(input_dim=16, hidden_dims=[32, 16], output_dim=5)
    pred_tactic, logits, conf = gate.forward(np.array([1.0, 2.0, 3.0]), m_res2["partial_state_vector"])
    assert 0 <= pred_tactic < 5
    assert 0.0 <= conf <= 1.0

    # 4. Graduated enforcement
    enf = GraduatedEnforcementPlane(shadow_mode=True)
    action = enf.determine_action(conf, True, 101, 1)
    assert action in (EnforcementAction.QUARANTINE_CGROUP, EnforcementAction.KILL_PROCESS)

    # 5. Self-protection
    meta = SelfProtectionMetaMonitor(authorized_loader_pid=1001)
    assert meta.mediate_bpf_syscall(1001, 'BPF_PROG_LOAD')
    assert not meta.mediate_bpf_syscall(9999, 'BPF_PROG_LOAD')

def test_bpf_codegen():
    spec = ArchitectureSpec(input_dim=16, hidden_dims=[32, 16], output_dim=2)
    c_code = generate_bpf_c_source(spec)
    assert "vital_gate_entry" in c_code
    assert "param_map" in c_code
    assert "monitor_map" in c_code
    assert "#pragma unroll" in c_code

def test_detection_benchmark():
    gen = ProvenanceTraceGenerator(seed=123)
    dataset = gen.generate_benchmark_dataset(samples_per_scenario=50, attack_ratio=0.3)
    detector = EBPFVitalDetector()
    results = evaluate_detector_on_scenarios(detector, dataset)
    assert results["Avg F1"] > 0.85
    assert results["FPR (%)"] < 5.0
