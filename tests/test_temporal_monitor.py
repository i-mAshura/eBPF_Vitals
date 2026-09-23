import pytest
import time
from ebpf_vital.kernel.temporal_monitor import (
    TemporalEvent,
    SubformulaStatus,
    TemporalPropertyDAG,
    BudgetScheduledMonitor,
    create_apt_exfiltration_dag
)

def test_exfiltration_dag_detection():
    dag = create_apt_exfiltration_dag()

    # Step 1: Benign read
    e0 = TemporalEvent(timestamp=100.0, event_id=0, op='read', src_obj=1, dst_obj=2, is_sensitive=False)
    st, rob = dag.evaluate_event(e0)
    assert st == SubformulaStatus.FALSIFIED

    # Step 2: Sensitive read
    e1 = TemporalEvent(timestamp=105.0, event_id=1, op='read', src_obj=1, dst_obj=3, is_sensitive=True)
    st, rob = dag.evaluate_event(e1)
    # Exfiltration not complete yet because send has not happened
    assert st == SubformulaStatus.FALSIFIED

    # Step 3: Network send to non-allowlisted destination within 30s (e.g. at 115s)
    e2 = TemporalEvent(timestamp=115.0, event_id=2, op='send', src_obj=1, dst_obj=4, is_allow_dest=False)
    st, rob = dag.evaluate_event(e2)
    # Exfiltration pattern is satisfied!
    assert st == SubformulaStatus.SATISFIED
    assert rob > 0.0

def test_budget_scheduled_monitor():
    monitor = BudgetScheduledMonitor(per_event_budget=50)
    dag1 = create_apt_exfiltration_dag() # cost 40
    dag2 = TemporalPropertyDAG(property_id=2, name="extra_prop", weight=0.5, horizon=10.0, cost=30)
    monitor.add_property(dag1)
    monitor.add_property(dag2)

    ev = TemporalEvent(timestamp=1.0, event_id=1, op='read', src_obj=1, dst_obj=2)
    res = monitor.evaluate(ev)
    # dag1 fits within budget 50 (spent 40)
    assert res["verdicts"][1] != SubformulaStatus.UNKNOWN
    # dag2 would exceed 50 (40 + 30 = 70 > 50) -> marked UNKNOWN
    assert res["verdicts"][2] == SubformulaStatus.UNKNOWN
