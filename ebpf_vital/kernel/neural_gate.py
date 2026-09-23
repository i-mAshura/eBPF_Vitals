r"""
In-Kernel Integer Neural Gate.
Coupled with the temporal monitor within the same eBPF tail-call chain.
Performs INT8 matrix multiplications and power-of-two shifts directly on kernel-resident state.
Ref: Section 8 of eBPF-VITAL paper.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np

class InKernelNeuralGate:
    """
    Fixed-point integer neural classifier running inside eBPF.
    All operations use 64-bit integer registers, signed multipliers, and ARSH scaling.
    """
    def __init__(self,
                 input_dim: int = 16,
                 hidden_dims: List[int] = None,
                 output_dim: int = 5,
                 scale_bits: int = 8):
        if hidden_dims is None:
            hidden_dims = [32, 16]
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.scale_bits = scale_bits

        # Initialize quantized integer weights in range [-127, 127]
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        prev_d = input_dim
        for h_d in hidden_dims + [output_dim]:
            w = np.random.randint(-64, 64, size=(h_d, prev_d), dtype=np.int32)
            b = np.random.randint(-32, 32, size=(h_d,), dtype=np.int32)
            self.weights.append(w)
            self.biases.append(b)
            prev_d = h_d

    def forward(self, input_features: np.ndarray, partial_monitor_state: List[int]) -> Tuple[int, np.ndarray, float]:
        """
        In-kernel inference step:
        Concatenates syscall features with partial subformula states (2-bit values 0,1,2).
        Returns: (predicted_tactic_class, logits_array, anomaly_confidence)
        """
        # Ensure fixed input dimension (pad or truncate)
        combined_features = np.zeros(self.input_dim, dtype=np.int32)
        n_feat = min(len(input_features), self.input_dim - len(partial_monitor_state))
        combined_features[:n_feat] = np.clip(input_features[:n_feat], -128, 127)

        # Append partial monitor states into remaining slots
        for i, val in enumerate(partial_monitor_state[:(self.input_dim - n_feat)]):
            # Scale 2-bit state (0: pending, 1: satisfied, 2: falsified) to integer activation
            combined_features[n_feat + i] = (val - 1) * 64

        # Run layers using pure integer arithmetic (matching eBPF bytecode instructions)
        activations = combined_features
        for l_idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            # 1. Multiply and accumulate in 64-bit integer
            acc = np.matmul(w, activations, dtype=np.int64) + b

            # 2. Power-of-two arithmetic right shift (simulates ARSH in eBPF)
            acc = acc >> self.scale_bits

            # 3. Activation: saturating piecewise linear ReLU
            if l_idx < len(self.weights) - 1:
                acc = np.maximum(0, acc)
                # Saturate at 127 (INT8 max)
                acc = np.minimum(127, acc)

            activations = acc.astype(np.int32)

        # Output logits
        logits = activations
        pred_class = int(np.argmax(logits))
        max_logit = float(logits[pred_class])
        # Approximate confidence in [0, 1]
        confidence = float(1.0 / (1.0 + np.exp(-max_logit / 32.0)))

        return pred_class, logits, confidence

    def update_weights(self, new_weights: List[np.ndarray], new_biases: List[np.ndarray]):
        """Atomic update of BPF parameter map slots."""
        self.weights = [np.copy(w) for w in new_weights]
        self.biases = [np.copy(b) for b in new_biases]
