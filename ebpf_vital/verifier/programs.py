"""
Candidate eBPF Neural Detector Program Builder and Mutator.
Constructs BPF instruction streams for integer neural inference and temporal monitoring,
with explicit parameters for layer widths, unroll factors, stack vs map scratch space,
and tail-call partitioning.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import random
from .abstract_interpreter import BPFInstruction, FailureReason

@dataclass
class ArchitectureSpec:
    input_dim: int = 16
    hidden_dims: List[int] = None
    output_dim: int = 2
    quant_bits: int = 8
    unroll_factor: int = 2
    use_map_scratch: bool = True
    tail_calls: int = 1
    activation: str = "relu"  # relu, clamp, shift_relu

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [32, 16]

    def to_vector(self) -> List[float]:
        """Feature vector representation of the architecture encoding phi."""
        vec = [
            float(self.input_dim),
            float(len(self.hidden_dims)),
            float(self.hidden_dims[0] if len(self.hidden_dims) > 0 else 0),
            float(self.hidden_dims[1] if len(self.hidden_dims) > 1 else 0),
            float(self.hidden_dims[2] if len(self.hidden_dims) > 2 else 0),
            float(self.output_dim),
            float(self.quant_bits),
            float(self.unroll_factor),
            1.0 if self.use_map_scratch else 0.0,
            float(self.tail_calls),
            1.0 if self.activation == "relu" else 2.0
        ]
        return vec

def build_neural_bpf_program(spec: ArchitectureSpec) -> List[BPFInstruction]:
    """
    Synthesizes a realistic eBPF instruction stream implementing the neural detector.
    - Uses r1 as ctx / input pointer.
    - r10 as stack frame or r7 as map scratch buffer pointer.
    - Accumulates in r0 / r8 / r9.
    - Uses saturating piecewise linear shifts for activations.
    """
    insns: List[BPFInstruction] = []

    # Setup return register r0 = 0
    insns.append(BPFInstruction(op='MOV', dst=0, imm=0))

    dims = [spec.input_dim] + spec.hidden_dims + [spec.output_dim]
    current_in = spec.input_dim

    # Setup scratch pointer
    if spec.use_map_scratch:
        # Load map pointer into r7 with safe bounds (e.g. 2048 bytes map)
        insns.append(BPFInstruction(op='MOV', dst=7, imm=0)) # Simulated map pointer
        map_size = 2048
    else:
        # Using stack directly (limit 512 bytes!)
        map_size = 0

    total_stack_needed = 0

    for layer_idx, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        # If tail call split requested at layer boundary
        if spec.tail_calls > 1 and layer_idx > 0 and (layer_idx % (len(dims) // spec.tail_calls) == 0):
            insns.append(BPFInstruction(op='TAIL_CALL', dst=0, imm=layer_idx))

        # Check stack consumption if not using map scratch
        if not spec.use_map_scratch:
            total_stack_needed += out_dim * 8
            if total_stack_needed > 500:
                # Deliberately emits stack store that triggers verifier overflow
                insns.append(BPFInstruction(op='MOV', dst=2, imm=42))
                insns.append(BPFInstruction(op='STX', dst=10, src=2, off=-(total_stack_needed + 20)))

        # Outer loop over output neurons
        for out_n in range(out_dim):
            # Accumulator in r8
            insns.append(BPFInstruction(op='MOV', dst=8, imm=0)) # bias

            # Inner loop over input neurons with unroll factor
            step = max(1, spec.unroll_factor)
            for in_n in range(0, in_dim, step):
                for u in range(min(step, in_dim - in_n)):
                    # Load simulated input from stack or ctx
                    insns.append(BPFInstruction(op='MOV', dst=9, imm=(in_n + u + 1) * 3))
                    # Multiply weight (simulated quantized weight in range [-127, 127])
                    weight_val = ((out_n * 7 + in_n * 13) % 255) - 128
                    insns.append(BPFInstruction(op='MOV', dst=3, imm=weight_val))
                    insns.append(BPFInstruction(op='MUL', dst=3, src=9))
                    # Accumulate
                    insns.append(BPFInstruction(op='ADD', dst=8, src=3))

            # Quantized fixed-point scaling shift (e.g. shift right by 7 or 8 bits)
            shift_amt = spec.quant_bits
            insns.append(BPFInstruction(op='MOV', dst=4, imm=shift_amt))
            insns.append(BPFInstruction(op='ARSH', dst=8, src=4))

            # Activation: piecewise linear ReLU / clamp
            if spec.activation in ("relu", "shift_relu"):
                # if r8 < 0 => r8 = 0
                skip_target = len(insns) + 2
                insns.append(BPFInstruction(op='JGE', dst=8, imm=0, target=skip_target))
                insns.append(BPFInstruction(op='MOV', dst=8, imm=0))

            # Store neuron activation in stack or scratch map
            if spec.use_map_scratch:
                stack_off = -((out_n % 16 + 1) * 8)
                insns.append(BPFInstruction(op='STX', dst=10, src=8, off=stack_off))
            else:
                stack_off = -(out_n * 8 + 8)
                if abs(stack_off) < 512:
                    insns.append(BPFInstruction(op='STX', dst=10, src=8, off=stack_off))

    # Final classification decision score into r0
    insns.append(BPFInstruction(op='MOV', dst=0, src=8))
    insns.append(BPFInstruction(op='EXIT', dst=0))

    return insns

def mutate_architecture(spec: ArchitectureSpec, failure_reason: Optional[FailureReason] = None) -> ArchitectureSpec:
    """
    Mutates architecture spec with gradient/reason-directed or random perturbations.
    """
    new_spec = ArchitectureSpec(
        input_dim=spec.input_dim,
        hidden_dims=list(spec.hidden_dims),
        output_dim=spec.output_dim,
        quant_bits=spec.quant_bits,
        unroll_factor=spec.unroll_factor,
        use_map_scratch=spec.use_map_scratch,
        tail_calls=spec.tail_calls,
        activation=spec.activation
    )

    if failure_reason == FailureReason.STACK_OVERFLOW:
        new_spec.use_map_scratch = True
        new_spec.hidden_dims = [max(8, w // 2) for w in new_spec.hidden_dims]

    elif failure_reason == FailureReason.COMPLEXITY_BUDGET:
        # Reduce unroll factor or split across tail-calls or reduce width
        if new_spec.unroll_factor > 1:
            new_spec.unroll_factor = max(1, new_spec.unroll_factor // 2)
        elif new_spec.tail_calls < 4:
            new_spec.tail_calls += 1
        else:
            new_spec.hidden_dims = [max(8, int(w * 0.75)) for w in new_spec.hidden_dims]

    elif failure_reason in (FailureReason.POINTER_OOB, FailureReason.SCALAR_RANGE_VIOLATION):
        new_spec.quant_bits = 8
        new_spec.use_map_scratch = True

    else:
        # Random small perturbation
        mutation_type = random.choice(['width', 'layer', 'unroll', 'quant'])
        if mutation_type == 'width' and new_spec.hidden_dims:
            idx = random.randint(0, len(new_spec.hidden_dims) - 1)
            delta = random.choice([-8, -4, 4, 8])
            new_spec.hidden_dims[idx] = max(8, min(128, new_spec.hidden_dims[idx] + delta))
        elif mutation_type == 'layer':
            if len(new_spec.hidden_dims) < 4 and random.random() > 0.5:
                new_spec.hidden_dims.append(random.choice([16, 32]))
            elif len(new_spec.hidden_dims) > 1:
                new_spec.hidden_dims.pop()
        elif mutation_type == 'unroll':
            new_spec.unroll_factor = random.choice([1, 2, 4])
        elif mutation_type == 'quant':
            new_spec.quant_bits = random.choice([8, 16])

    return new_spec
