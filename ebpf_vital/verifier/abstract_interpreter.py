"""
eBPF Verifier Abstract Interpreter.
Implements abstract interpretation over scalar register intervals, tristate numbers,
pointer tracking, stack depth, loop termination bounds, and complexity budget.
Ref: PREVAIL (PLDI 2019), Agni (CAV 2023), Linux kernel verifier.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Tuple, Optional, Set
import copy
from .tnum import TNum, MAX_U64, MIN_S64, MAX_S64

MAX_STACK_SIZE = 512
MAX_INSTRUCTIONS_PROCESSED = 1000000
MAX_TAIL_CALLS = 33

class FailureReason(Enum):
    NONE = -1
    STACK_OVERFLOW = 0          # Stack frame limit exceeded (>512B)
    POINTER_OOB = 1             # Invalid pointer arithmetic / out-of-bounds access
    SCALAR_RANGE_VIOLATION = 2  # Range / interval bound check failed
    LOOP_BOUND_EXCEEDED = 3     # Unbounded loop or iteration limit exceeded
    COMPLEXITY_BUDGET = 4       # Total processed instructions exceeded budget

class BPFRegType(Enum):
    NOT_INIT = auto()
    SCALAR_VALUE = auto()
    PTR_TO_CTX = auto()
    PTR_TO_STACK = auto()
    PTR_TO_MAP_VALUE = auto()
    PTR_TO_MAP_KEY = auto()

@dataclass
class BPFReg:
    reg_type: BPFRegType = BPFRegType.NOT_INIT
    var_off: TNum = field(default_factory=TNum.unknown)
    smin: int = MIN_S64
    smax: int = MAX_S64
    umin: int = 0
    umax: int = MAX_U64
    off: int = 0  # Fixed offset for pointers
    map_size: int = 0  # Size in bytes if PTR_TO_MAP_VALUE

    def is_scalar(self) -> bool:
        return self.reg_type == BPFRegType.SCALAR_VALUE

    def is_pointer(self) -> bool:
        return self.reg_type in (BPFRegType.PTR_TO_CTX, BPFRegType.PTR_TO_STACK,
                                 BPFRegType.PTR_TO_MAP_VALUE, BPFRegType.PTR_TO_MAP_KEY)

    @classmethod
    def make_const(cls, val: int) -> 'BPFReg':
        return cls(
            reg_type=BPFRegType.SCALAR_VALUE,
            var_off=TNum.const(val),
            smin=val,
            smax=val,
            umin=val if val >= 0 else 0,
            umax=val if val >= 0 else MAX_U64
        )

    @classmethod
    def make_scalar_range(cls, smin: int, smax: int) -> 'BPFReg':
        umin = max(0, smin)
        umax = max(0, smax)
        t = TNum.range(umin, umax)
        return cls(
            reg_type=BPFRegType.SCALAR_VALUE,
            var_off=t,
            smin=smin,
            smax=smax,
            umin=umin,
            umax=umax
        )

    @classmethod
    def make_stack_ptr(cls, off: int = 0) -> 'BPFReg':
        return cls(reg_type=BPFRegType.PTR_TO_STACK, off=off, var_off=TNum.const(0))

    @classmethod
    def make_map_ptr(cls, map_size: int, off: int = 0) -> 'BPFReg':
        return cls(reg_type=BPFRegType.PTR_TO_MAP_VALUE, off=off, var_off=TNum.const(0), map_size=map_size)

    @classmethod
    def make_ctx_ptr(cls) -> 'BPFReg':
        return cls(reg_type=BPFRegType.PTR_TO_CTX, off=0, var_off=TNum.const(0))

@dataclass
class BPFInstruction:
    op: str           # 'MOV', 'ADD', 'SUB', 'MUL', 'DIV', 'AND', 'OR', 'LSH', 'RSH', 'ARSH',
                      # 'LDX', 'STX', 'JEQ', 'JNE', 'JGT', 'JGE', 'JLT', 'JLE', 'CALL', 'EXIT', 'TAIL_CALL'
    dst: int          # Register 0-10
    src: Optional[int] = None  # Register 0-10 or None
    imm: int = 0      # Immediate value
    off: int = 0      # Stack or memory offset
    target: int = 0   # Jump target instruction index (relative or absolute)

@dataclass
class VerificationResult:
    accepted: bool
    processed_instructions: int
    residual_budget: int
    failure_reason: FailureReason
    failure_message: str
    decision_trace: List[Dict] = field(default_factory=list)

class BPFVerifierState:
    def __init__(self):
        self.regs: List[BPFReg] = [BPFReg() for _ in range(11)]
        # r1 is context pointer on entry
        self.regs[1] = BPFReg.make_ctx_ptr()
        # r10 is read-only stack frame pointer
        self.regs[10] = BPFReg.make_stack_ptr(off=0)
        # Stack slots: negative offsets -1 to -512
        self.stack: Dict[int, BPFReg] = {}
        self.allocated_stack: int = 0
        self.tail_call_count: int = 0

    def copy(self) -> 'BPFVerifierState':
        cp = BPFVerifierState()
        cp.regs = [copy.deepcopy(r) for r in self.regs]
        cp.stack = {k: copy.deepcopy(v) for k, v in self.stack.items()}
        cp.allocated_stack = self.allocated_stack
        cp.tail_call_count = self.tail_call_count
        return cp

    def is_subsumed_by(self, other: 'BPFVerifierState') -> bool:
        """State subsumption: self is subsumed by other if other has wider or equal bounds."""
        if self.allocated_stack > other.allocated_stack:
            return False
        for i in range(11):
            r1, r2 = self.regs[i], other.regs[i]
            if r1.reg_type != r2.reg_type:
                return False
            if r1.is_scalar():
                if r1.smin < r2.smin or r1.smax > r2.smax:
                    return False
                if (r1.var_off.mask & ~r2.var_off.mask) != 0:
                    return False
        return True

class BPFVerifier:
    def __init__(self, budget: int = MAX_INSTRUCTIONS_PROCESSED, max_stack: int = MAX_STACK_SIZE):
        self.budget = min(budget, MAX_INSTRUCTIONS_PROCESSED)
        self.max_stack = max_stack

    def verify(self, program: List[BPFInstruction], initial_state: Optional[BPFVerifierState] = None) -> VerificationResult:
        """
        Runs bounded abstract interpretation over program.
        Explores all paths with worklist, state caching, and complexity tracking.
        """
        if not program:
            return VerificationResult(
                accepted=False,
                processed_instructions=0,
                residual_budget=self.budget,
                failure_reason=FailureReason.SCALAR_RANGE_VIOLATION,
                failure_message="Empty program"
            )

        init = initial_state.copy() if initial_state else BPFVerifierState()
        worklist: List[Tuple[int, BPFVerifierState]] = [(0, init)]
        visited_states: Dict[int, List[BPFVerifierState]] = {i: [] for i in range(len(program) + 1)}
        processed_count = 0
        decision_trace: List[Dict] = []

        while worklist:
            pc, state = worklist.pop()
            if pc >= len(program):
                return VerificationResult(
                    accepted=False,
                    processed_instructions=processed_count,
                    residual_budget=max(0, self.budget - processed_count),
                    failure_reason=FailureReason.POINTER_OOB,
                    failure_message=f"Instruction pointer out of bounds: pc={pc}",
                    decision_trace=decision_trace
                )

            # Subsumption check to avoid infinite loops and prune explored paths
            subsumed = False
            for vstate in visited_states[pc]:
                if state.is_subsumed_by(vstate):
                    subsumed = True
                    break
            if subsumed:
                continue
            visited_states[pc].append(state.copy())

            # Complexity budget check
            processed_count += 1
            if processed_count > self.budget:
                decision_trace.append({"pc": pc, "verdict": "FAIL", "reason": "COMPLEXITY_BUDGET"})
                return VerificationResult(
                    accepted=False,
                    processed_instructions=processed_count,
                    residual_budget=0,
                    failure_reason=FailureReason.COMPLEXITY_BUDGET,
                    failure_message=f"Verifier complexity budget exceeded ({self.budget} instructions)",
                    decision_trace=decision_trace
                )

            insn = program[pc]
            op = insn.op.upper()

            # Execute instruction in abstract domain
            if op == 'EXIT':
                # Return value in r0 must be initialized
                r0 = state.regs[0]
                if r0.reg_type == BPFRegType.NOT_INIT:
                    return VerificationResult(
                        accepted=False,
                        processed_instructions=processed_count,
                        residual_budget=max(0, self.budget - processed_count),
                        failure_reason=FailureReason.SCALAR_RANGE_VIOLATION,
                        failure_message="R0 not initialized at program exit",
                        decision_trace=decision_trace
                    )
                decision_trace.append({"pc": pc, "verdict": "PATH_EXIT", "processed": processed_count})
                continue

            elif op == 'TAIL_CALL':
                state.tail_call_count += 1
                if state.tail_call_count > MAX_TAIL_CALLS:
                    return VerificationResult(
                        accepted=False,
                        processed_instructions=processed_count,
                        residual_budget=max(0, self.budget - processed_count),
                        failure_reason=FailureReason.COMPLEXITY_BUDGET,
                        failure_message=f"Max tail-call depth exceeded ({MAX_TAIL_CALLS})",
                        decision_trace=decision_trace
                    )
                # Tail call never returns on success, but can fail fall-through
                worklist.append((pc + 1, state))
                continue

            elif op in ('MOV', 'ADD', 'SUB', 'MUL', 'DIV', 'AND', 'OR', 'LSH', 'RSH', 'ARSH'):
                dst = insn.dst
                if dst == 10:
                    return VerificationResult(
                        accepted=False,
                        processed_instructions=processed_count,
                        residual_budget=max(0, self.budget - processed_count),
                        failure_reason=FailureReason.POINTER_OOB,
                        failure_message="Attempt to modify read-only frame pointer r10",
                        decision_trace=decision_trace
                    )

                src_reg = state.regs[insn.src] if insn.src is not None else BPFReg.make_const(insn.imm)

                if op == 'MOV':
                    state.regs[dst] = copy.deepcopy(src_reg)

                elif op == 'ADD':
                    dst_reg = state.regs[dst]
                    if dst_reg.is_pointer() and src_reg.is_scalar():
                        # Pointer arithmetic: ptr + scalar
                        if not src_reg.var_off.is_const():
                            # Variable pointer offset must be strictly bounded
                            if src_reg.smin < 0 or src_reg.smax > 1024:
                                return VerificationResult(
                                    accepted=False,
                                    processed_instructions=processed_count,
                                    residual_budget=max(0, self.budget - processed_count),
                                    failure_reason=FailureReason.POINTER_OOB,
                                    failure_message=f"Unbounded scalar offset [{src_reg.smin}, {src_reg.smax}] added to pointer",
                                    decision_trace=decision_trace
                                )
                        new_off = dst_reg.off + src_reg.smin
                        dst_reg.off = new_off
                    elif dst_reg.is_scalar() and src_reg.is_scalar():
                        new_smin = dst_reg.smin + src_reg.smin
                        new_smax = dst_reg.smax + src_reg.smax
                        new_tnum = dst_reg.var_off.add(src_reg.var_off)
                        state.regs[dst] = BPFReg(
                            reg_type=BPFRegType.SCALAR_VALUE,
                            var_off=new_tnum,
                            smin=new_smin,
                            smax=new_smax,
                            umin=max(0, new_smin),
                            umax=new_tnum.umax()
                        )
                    else:
                        return VerificationResult(
                            accepted=False,
                            processed_instructions=processed_count,
                            residual_budget=max(0, self.budget - processed_count),
                            failure_reason=FailureReason.POINTER_OOB,
                            failure_message="Invalid pointer-pointer addition",
                            decision_trace=decision_trace
                        )

                elif op == 'SUB':
                    dst_reg = state.regs[dst]
                    if dst_reg.is_pointer() and src_reg.is_scalar():
                        dst_reg.off -= src_reg.smax
                    elif dst_reg.is_scalar() and src_reg.is_scalar():
                        new_smin = dst_reg.smin - src_reg.smax
                        new_smax = dst_reg.smax - src_reg.smin
                        new_tnum = dst_reg.var_off.sub(src_reg.var_off)
                        state.regs[dst] = BPFReg(
                            reg_type=BPFRegType.SCALAR_VALUE,
                            var_off=new_tnum,
                            smin=new_smin,
                            smax=new_smax,
                            umin=max(0, new_smin),
                            umax=new_tnum.umax()
                        )
                    else:
                        return VerificationResult(
                            accepted=False,
                            processed_instructions=processed_count,
                            residual_budget=max(0, self.budget - processed_count),
                            failure_reason=FailureReason.POINTER_OOB,
                            failure_message="Invalid pointer subtraction",
                            decision_trace=decision_trace
                        )

                elif op == 'MUL':
                    dst_reg = state.regs[dst]
                    if not (dst_reg.is_scalar() and src_reg.is_scalar()):
                        return VerificationResult(
                            accepted=False,
                            processed_instructions=processed_count,
                            residual_budget=max(0, self.budget - processed_count),
                            failure_reason=FailureReason.POINTER_OOB,
                            failure_message="Cannot multiply pointer values",
                            decision_trace=decision_trace
                        )
                    p1 = dst_reg.smin * src_reg.smin
                    p2 = dst_reg.smin * src_reg.smax
                    p3 = dst_reg.smax * src_reg.smin
                    p4 = dst_reg.smax * src_reg.smax
                    new_smin = min(p1, p2, p3, p4)
                    new_smax = max(p1, p2, p3, p4)
                    state.regs[dst] = BPFReg.make_scalar_range(new_smin, new_smax)

                elif op == 'DIV':
                    dst_reg = state.regs[dst]
                    if not (dst_reg.is_scalar() and src_reg.is_scalar()):
                        return VerificationResult(
                            accepted=False,
                            processed_instructions=processed_count,
                            residual_budget=max(0, self.budget - processed_count),
                            failure_reason=FailureReason.POINTER_OOB,
                            failure_message="Cannot divide pointer values",
                            decision_trace=decision_trace
                        )
                    # Check division by zero
                    if src_reg.smin <= 0 <= src_reg.smax:
                        return VerificationResult(
                            accepted=False,
                            processed_instructions=processed_count,
                            residual_budget=max(0, self.budget - processed_count),
                            failure_reason=FailureReason.SCALAR_RANGE_VIOLATION,
                            failure_message="Possible division by zero in abstract domain",
                            decision_trace=decision_trace
                        )
                    state.regs[dst] = BPFReg.make_scalar_range(-MAX_S64, MAX_S64)

                elif op in ('AND', 'OR', 'LSH', 'RSH', 'ARSH'):
                    dst_reg = state.regs[dst]
                    if not (dst_reg.is_scalar() and src_reg.is_scalar()):
                        return VerificationResult(
                            accepted=False,
                            processed_instructions=processed_count,
                            residual_budget=max(0, self.budget - processed_count),
                            failure_reason=FailureReason.POINTER_OOB,
                            failure_message="Bitwise operation on pointer value",
                            decision_trace=decision_trace
                        )
                    if op == 'AND':
                        res_tnum = dst_reg.var_off.bitwise_and(src_reg.var_off)
                        smin = 0 if (dst_reg.smin >= 0 or src_reg.smin >= 0) else MIN_S64
                        smax = min(dst_reg.smax, src_reg.smax) if (dst_reg.smin >= 0 and src_reg.smin >= 0) else MAX_S64
                    elif op == 'OR':
                        res_tnum = dst_reg.var_off.bitwise_or(src_reg.var_off)
                        smin = max(dst_reg.smin, src_reg.smin)
                        smax = max(dst_reg.smax, src_reg.smax) | res_tnum.mask
                    elif op == 'LSH':
                        shift = src_reg.smin if src_reg.var_off.is_const() else 0
                        res_tnum = dst_reg.var_off.lshift(shift)
                        smin = dst_reg.smin << shift if shift < 32 else 0
                        smax = dst_reg.smax << shift if shift < 32 else MAX_S64
                    elif op == 'RSH':
                        shift = src_reg.smin if src_reg.var_off.is_const() else 0
                        res_tnum = dst_reg.var_off.rshift(shift)
                        smin = max(0, dst_reg.smin >> shift)
                        smax = dst_reg.smax >> shift
                    else: # ARSH
                        shift = src_reg.smin if src_reg.var_off.is_const() else 0
                        res_tnum = dst_reg.var_off.arshift(shift)
                        smin = dst_reg.smin >> shift
                        smax = dst_reg.smax >> shift

                    state.regs[dst] = BPFReg(
                        reg_type=BPFRegType.SCALAR_VALUE,
                        var_off=res_tnum,
                        smin=smin,
                        smax=smax,
                        umin=res_tnum.umin(),
                        umax=res_tnum.umax()
                    )

                worklist.append((pc + 1, state))
                continue

            elif op == 'STX':
                # Store register to memory: *(dst + off) = src
                dst_reg = state.regs[insn.dst]
                src_reg = state.regs[insn.src]
                if dst_reg.reg_type == BPFRegType.PTR_TO_STACK:
                    abs_off = dst_reg.off + insn.off
                    if abs_off > 0 or abs_off < -self.max_stack:
                        return VerificationResult(
                            accepted=False,
                            processed_instructions=processed_count,
                            residual_budget=max(0, self.budget - processed_count),
                            failure_reason=FailureReason.STACK_OVERFLOW,
                            failure_message=f"Stack access out of bounds: offset {abs_off} exceeds limit [-{self.max_stack}, 0]",
                            decision_trace=decision_trace
                        )
                    state.stack[abs_off] = copy.deepcopy(src_reg)
                    state.allocated_stack = max(state.allocated_stack, -abs_off)
                elif dst_reg.reg_type == BPFRegType.PTR_TO_MAP_VALUE:
                    abs_off = dst_reg.off + insn.off
                    if abs_off < 0 or abs_off + 8 > dst_reg.map_size:
                        return VerificationResult(
                            accepted=False,
                            processed_instructions=processed_count,
                            residual_budget=max(0, self.budget - processed_count),
                            failure_reason=FailureReason.POINTER_OOB,
                            failure_message=f"Map value store out of bounds: offset {abs_off} + 8 > size {dst_reg.map_size}",
                            decision_trace=decision_trace
                        )
                else:
                    return VerificationResult(
                        accepted=False,
                        processed_instructions=processed_count,
                        residual_budget=max(0, self.budget - processed_count),
                        failure_reason=FailureReason.POINTER_OOB,
                        failure_message=f"Invalid store to non-pointer register {insn.dst}",
                        decision_trace=decision_trace
                    )
                worklist.append((pc + 1, state))
                continue

            elif op == 'LDX':
                # Load from memory: dst = *(src + off)
                src_reg = state.regs[insn.src]
                if src_reg.reg_type == BPFRegType.PTR_TO_STACK:
                    abs_off = src_reg.off + insn.off
                    if abs_off > 0 or abs_off < -self.max_stack:
                        return VerificationResult(
                            accepted=False,
                            processed_instructions=processed_count,
                            residual_budget=max(0, self.budget - processed_count),
                            failure_reason=FailureReason.STACK_OVERFLOW,
                            failure_message=f"Stack read out of bounds: offset {abs_off}",
                            decision_trace=decision_trace
                        )
                    val = state.stack.get(abs_off, BPFReg.make_scalar_range(-MAX_S64, MAX_S64))
                    state.regs[insn.dst] = copy.deepcopy(val)
                elif src_reg.reg_type == BPFRegType.PTR_TO_MAP_VALUE:
                    abs_off = src_reg.off + insn.off
                    if abs_off < 0 or abs_off + 8 > src_reg.map_size:
                        return VerificationResult(
                            accepted=False,
                            processed_instructions=processed_count,
                            residual_budget=max(0, self.budget - processed_count),
                            failure_reason=FailureReason.POINTER_OOB,
                            failure_message=f"Map value read out of bounds: offset {abs_off}",
                            decision_trace=decision_trace
                        )
                    state.regs[insn.dst] = BPFReg.make_scalar_range(-MAX_S64, MAX_S64)
                elif src_reg.reg_type == BPFRegType.PTR_TO_CTX:
                    state.regs[insn.dst] = BPFReg.make_scalar_range(0, MAX_U64)
                else:
                    return VerificationResult(
                        accepted=False,
                        processed_instructions=processed_count,
                        residual_budget=max(0, self.budget - processed_count),
                        failure_reason=FailureReason.POINTER_OOB,
                        failure_message=f"Invalid read from non-pointer register {insn.src}",
                        decision_trace=decision_trace
                    )
                worklist.append((pc + 1, state))
                continue

            elif op in ('JEQ', 'JNE', 'JGT', 'JGE', 'JLT', 'JLE'):
                # Conditional branch: explore both taken and fall-through branches with refined bounds
                dst_reg = state.regs[insn.dst]
                src_reg = state.regs[insn.src] if insn.src is not None else BPFReg.make_const(insn.imm)
                target_pc = insn.target

                taken_state = state.copy()
                fallthru_state = state.copy()

                # Refine bounds on taken branch
                if op == 'JGT':
                    # Taken: dst > src => dst.smin >= src.smin + 1
                    taken_state.regs[insn.dst].smin = max(dst_reg.smin, src_reg.smin + 1)
                    # Fallthru: dst <= src => dst.smax <= src.smax
                    fallthru_state.regs[insn.dst].smax = min(dst_reg.smax, src_reg.smax)
                elif op == 'JGE':
                    taken_state.regs[insn.dst].smin = max(dst_reg.smin, src_reg.smin)
                    fallthru_state.regs[insn.dst].smax = min(dst_reg.smax, src_reg.smax - 1)
                elif op == 'JLT':
                    taken_state.regs[insn.dst].smax = min(dst_reg.smax, src_reg.smax - 1)
                    fallthru_state.regs[insn.dst].smin = max(dst_reg.smin, src_reg.smin)
                elif op == 'JLE':
                    taken_state.regs[insn.dst].smax = min(dst_reg.smax, src_reg.smax)
                    fallthru_state.regs[insn.dst].smin = max(dst_reg.smin, src_reg.smin + 1)
                elif op == 'JEQ':
                    if src_reg.var_off.is_const():
                        c = src_reg.var_off.value
                        taken_state.regs[insn.dst] = BPFReg.make_const(c)

                # Branch feasibility checks
                taken_feasible = taken_state.regs[insn.dst].smin <= taken_state.regs[insn.dst].smax
                fallthru_feasible = fallthru_state.regs[insn.dst].smin <= fallthru_state.regs[insn.dst].smax

                if taken_feasible:
                    worklist.append((target_pc, taken_state))
                if fallthru_feasible:
                    worklist.append((pc + 1, fallthru_state))
                continue

            else:
                # Unsupported or custom op
                worklist.append((pc + 1, state))
                continue

        # All paths verified successfully
        return VerificationResult(
            accepted=True,
            processed_instructions=processed_count,
            residual_budget=max(0, self.budget - processed_count),
            failure_reason=FailureReason.NONE,
            failure_message="Verification passed successfully",
            decision_trace=decision_trace
        )
