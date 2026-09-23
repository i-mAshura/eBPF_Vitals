"""
Tristate number (tnum) implementation matching the Linux kernel eBPF verifier.
Ref: Vishwanathan et al., "Sound, Precise, and Fast Abstract Interpretation with Tristate Numbers", CGO 2022.
Linux kernel: include/linux/tnum.h, kernel/bpf/tnum.c
"""

from dataclasses import dataclass
from typing import Tuple

MASK_64 = 0xFFFFFFFFFFFFFFFF
SIGN_BIT_64 = 0x8000000000000000
MAX_U64 = 0xFFFFFFFFFFFFFFFF
MIN_S64 = -0x8000000000000000
MAX_S64 = 0x7FFFFFFFFFFFFFFF

@dataclass(frozen=True)
class TNum:
    value: int  # Known 1s (where mask is 0)
    mask: int   # Unknown bits (1 = unknown, 0 = known)

    def __post_init__(self):
        v = self.value & MASK_64
        m = self.mask & MASK_64
        # Value must have 0 where mask has 1
        object.__setattr__(self, 'value', v & ~m)
        object.__setattr__(self, 'mask', m)

    @classmethod
    def const(cls, v: int) -> 'TNum':
        return cls(v & MASK_64, 0)

    @classmethod
    def unknown(cls) -> 'TNum':
        return cls(0, MASK_64)

    @classmethod
    def range(cls, min_val: int, max_val: int) -> 'TNum':
        """Construct tnum containing all values in [min_val, max_val]."""
        min_v = max(0, min_val) & MASK_64
        max_v = max(0, max_val) & MASK_64
        if min_v > max_v:
            return cls.unknown()
        diff = min_v ^ max_v
        if diff == 0:
            return cls.const(min_v)
        # Find highest set bit of diff
        top = 1 << (diff.bit_length() - 1)
        mask = (top * 2 - 1) & MASK_64
        return cls(min_v & ~mask, mask)

    def is_const(self) -> bool:
        return self.mask == 0

    def is_unknown(self) -> bool:
        return self.mask == MASK_64

    def umin(self) -> int:
        return self.value

    def umax(self) -> int:
        return self.value | self.mask

    def smin(self) -> int:
        """Signed minimum value."""
        if self.mask & SIGN_BIT_64:
            # Sign bit is unknown: smallest is negative with sign bit set
            val = (self.value | SIGN_BIT_64)
            # Make negative by extending sign
            s = val if val < 0x8000000000000000 else val - (1 << 64)
            return s
        else:
            # Sign bit is known
            if self.value & SIGN_BIT_64:
                # Known negative
                return self.value - (1 << 64)
            else:
                # Known positive
                return self.value

    def smax(self) -> int:
        """Signed maximum value."""
        if self.mask & SIGN_BIT_64:
            # Sign bit is unknown: max is positive with sign bit 0
            return (self.value | (self.mask & ~SIGN_BIT_64))
        else:
            if self.value & SIGN_BIT_64:
                return (self.value | self.mask) - (1 << 64)
            else:
                return (self.value | self.mask)

    def add(self, other: 'TNum') -> 'TNum':
        v = (self.value + other.value) & MASK_64
        m = (self.mask + other.mask) & MASK_64
        # Any carry ripples through unknown bits
        alpha = 2 * (self.mask | other.mask) & MASK_64
        res_mask = (m | alpha) & MASK_64
        return TNum(v & ~res_mask, res_mask)

    def sub(self, other: 'TNum') -> 'TNum':
        # a - b = a + (~b + 1)
        v = (self.value - other.value) & MASK_64
        m = (self.mask + other.mask) & MASK_64
        alpha = 2 * (self.mask | other.mask) & MASK_64
        res_mask = (m | alpha) & MASK_64
        return TNum(v & ~res_mask, res_mask)

    def bitwise_and(self, other: 'TNum') -> 'TNum':
        # 1 & 1 = 1; 0 & x = 0; ? & 1 = ?; ? & ? = ?
        v = self.value & other.value
        m = (self.mask & (other.value | other.mask)) | (other.mask & (self.value | self.mask))
        return TNum(v & ~m, m)

    def bitwise_or(self, other: 'TNum') -> 'TNum':
        v = self.value | other.value
        m = self.mask | other.mask
        return TNum(v & ~m, m)

    def bitwise_xor(self, other: 'TNum') -> 'TNum':
        v = self.value ^ other.value
        m = self.mask | other.mask
        return TNum(v & ~m, m)

    def lshift(self, shift: int) -> 'TNum':
        if shift >= 64:
            return TNum.const(0)
        v = (self.value << shift) & MASK_64
        m = (self.mask << shift) & MASK_64
        return TNum(v & ~m, m)

    def rshift(self, shift: int) -> 'TNum':
        if shift >= 64:
            return TNum.const(0)
        v = (self.value >> shift) & MASK_64
        m = (self.mask >> shift) & MASK_64
        return TNum(v & ~m, m)

    def arshift(self, shift: int) -> 'TNum':
        """Arithmetic right shift."""
        if shift >= 64:
            shift = 63
        sign = bool(self.value & SIGN_BIT_64)
        sign_unknown = bool(self.mask & SIGN_BIT_64)
        v = (self.value >> shift)
        m = (self.mask >> shift)
        if sign_unknown:
            ext_mask = ((1 << shift) - 1) << (64 - shift)
            m |= ext_mask
        elif sign:
            ext_val = ((1 << shift) - 1) << (64 - shift)
            v |= ext_val
        return TNum(v & MASK_64, m & MASK_64)

    def intersect(self, other: 'TNum') -> 'TNum':
        """Intersection of two tnums; if contradictory, returns empty/infeasible."""
        common_known = ~(self.mask | other.mask) & MASK_64
        if (self.value ^ other.value) & common_known:
            # Contradiction: bit is known 1 in one and known 0 in other
            return TNum(0, 0) # Infeasible
        v = (self.value | other.value) & MASK_64
        m = (self.mask & other.mask) & MASK_64
        return TNum(v & ~m, m)

    def __repr__(self) -> str:
        chars = []
        for i in range(63, -1, -1):
            bit = 1 << i
            if self.mask & bit:
                chars.append('?')
            elif self.value & bit:
                chars.append('1')
            else:
                chars.append('0')
        return f"TNum({''.join(chars[:16])}... [u: {self.umin()}, {self.umax()}])"
