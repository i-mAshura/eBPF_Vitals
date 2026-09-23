import pytest
from ebpf_vital.verifier.tnum import TNum
from ebpf_vital.verifier.abstract_interpreter import (
    BPFVerifier,
    BPFInstruction,
    BPFReg,
    BPFVerifierState,
    FailureReason
)
from ebpf_vital.verifier.programs import ArchitectureSpec, build_neural_bpf_program

def test_tnum_constants_and_ranges():
    c5 = TNum.const(5)
    assert c5.is_const()
    assert c5.umin() == 5
    assert c5.umax() == 5
    assert c5.smin() == 5
    assert c5.smax() == 5

    r = TNum.range(10, 20)
    assert r.umin() <= 10
    assert r.umax() >= 20
    assert not r.is_const()

def test_tnum_arithmetic():
    a = TNum.const(15)
    b = TNum.const(7)
    add_res = a.add(b)
    assert add_res.is_const()
    assert add_res.umin() == 22

    sub_res = a.sub(b)
    assert sub_res.is_const()
    assert sub_res.umin() == 8

    and_res = a.bitwise_and(b)
    assert and_res.is_const()
    assert and_res.umin() == 7

    shift_res = b.lshift(2)
    assert shift_res.is_const()
    assert shift_res.umin() == 28

def test_verifier_accept_valid_program():
    verifier = BPFVerifier(budget=50000)
    spec = ArchitectureSpec(input_dim=8, hidden_dims=[16, 8], output_dim=2, unroll_factor=1, use_map_scratch=True)
    prog = build_neural_bpf_program(spec)
    res = verifier.verify(prog)
    assert res.accepted
    assert res.failure_reason == FailureReason.NONE
    assert res.processed_instructions > 0
    assert res.residual_budget > 0

def test_verifier_stack_overflow_detection():
    verifier = BPFVerifier(budget=50000, max_stack=512)
    # Deliberately build program without map scratch that allocates too much stack
    spec = ArchitectureSpec(input_dim=16, hidden_dims=[64, 64], output_dim=2, unroll_factor=1, use_map_scratch=False)
    prog = build_neural_bpf_program(spec)
    res = verifier.verify(prog)
    assert not res.accepted
    assert res.failure_reason == FailureReason.STACK_OVERFLOW
    assert "Stack" in res.failure_message

def test_verifier_complexity_budget_exceeded():
    # Tight budget of only 50 instructions
    verifier = BPFVerifier(budget=50)
    spec = ArchitectureSpec(input_dim=16, hidden_dims=[32, 16], output_dim=2)
    prog = build_neural_bpf_program(spec)
    res = verifier.verify(prog)
    assert not res.accepted
    assert res.failure_reason == FailureReason.COMPLEXITY_BUDGET
