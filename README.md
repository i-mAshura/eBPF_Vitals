# eBPF-VITAL: Verifier-in-the-Loop Adaptation for Neurosymbolic Intrusion Detection

[![Tests](https://img.shields.io/badge/tests-18%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.14-blue.svg)]()
[![License](https://img.shields.io/badge/license-GPL--2.0%20%2F%20MIT-blue.svg)]()

Working prototype for **eBPF-VITAL** (*Verifier-in-the-Loop Adaptation for Neurosymbolic Intrusion Detection*). 

Instead of traditional *fit-first* approaches (shrinking and quantizing models until they happen to pass static verification), eBPF-VITAL treats the eBPF abstract-interpretation acceptance boundary as a **differentiable training signal** inside neural architecture search, distillation, and online continual adaptation.

---

## Architecture Overview

```
                                  +-------------------------------------+
                                  |      Userspace Control Plane        |
                                  |  - GNN Verifier Surrogate (\hat{V}) |
                                  |  - Verifier-Constrained NAS         |
                                  |  - Projected Online Updater (Alg 1) |
                                  |  - Two-Tier Checkpoints (Alg 2)     |
                                  +-------------------------------------+
                                                     |
                                   [Canary Checks / Atomic Map Commits]
                                                     v
+-----------------------------------------------------------------------------------+
|                            In-Kernel Data Plane                                   |
|                                                                                   |
|  +--------------------+     +-----------------------+     +--------------------+  |
|  | LSM Capture Plane  | --> | pt-LTL Monitor DAG    | --> | Integer Neural     |  |
|  | (Provenance/Taint) |     | (Budget Sched, Alg 3) |     | Gate (INT8 + ARSH) |  |
|  +--------------------+     +-----------------------+     +--------------------+  |
|                                                                     |             |
|                                                                     v             |
|  +--------------------+                                   +--------------------+  |
|  | Meta-Monitor       |                                   | Graduated          |  |
|  | (Watchdog/Heartbt) |                                   | Enforcement        |  |
|  +--------------------+                                   +--------------------+  |
+-----------------------------------------------------------------------------------+
```

---

## Directory Structure

```
.
├── ebpf_vital/
│   ├── verifier/
│   │   ├── tnum.py                 # 64-bit Tristate numbers (known bits/mask)
│   │   ├── abstract_interpreter.py # eBPF Abstract interpreter engine (PREVAIL-aligned)
│   │   └── programs.py             # Candidate BPF instruction synthesizer & mutator
│   ├── surrogate/
│   │   ├── program_graph.py        # Relational CFG, DFG, stack dependence extraction
│   │   ├── gnn_surrogate.py        # 3-head MPNN surrogate (\hat{V}_\theta)
│   │   ├── calibration.py          # Temperature scaling, ECE, Eq. (4) bound check
│   │   └── adversarial_mining.py   # Boundary program miner (\hat{V}_\theta \approx 0.5)
│   ├── nas/
│   │   ├── supernet.py             # Differentiable supernetwork with INT8 STE
│   │   ├── distillation.py         # Knowledge distillation & STL robustness loss
│   │   └── search.py               # Constrained search & target kernel reject loop
│   ├── adaptation/
│   │   ├── continual_updater.py    # Algorithm 1: Projected Online Parameter Update
│   │   └── checkpoint_protocol.py  # Algorithm 2: Two-Tier Checkpoint & Rollback Protocol
│   ├── kernel/
│   │   ├── temporal_monitor.py     # In-kernel pt-LTL/MTL subformula DAG (Algorithm 3)
│   │   ├── neural_gate.py          # INT8 neural classifier with ARSH bit-shift scaling
│   │   └── supporting_planes.py    # LSM provenance, graduated enforcement, watchdog
│   ├── bpf_codegen/
│   │   └── generator.py            # Compilable C / libbpf CO-RE program generator
│   └── evaluation/
│       ├── provenance_generator.py # DARPA TC (CADETS, THEIA, TRACE) & StreamSpot generator
│       ├── benchmarks.py           # Baseline detectors (Falco, Tracee, Tetragon, FLASH, etc.)
│       └── runner.py               # Benchmark execution engine
├── tests/                          # Full pytest test suite (18 tests)
├── vital_detector.bpf.c            # Synthesized in-kernel eBPF C source (CO-RE)
├── run_full_evaluation.py          # End-to-end evaluation runner script
└── research-proposal.txt           # Formal specification & proposal document
```

---

## Key Results & Verification Targets

### 1. Detection Quality (Table 1)
| System | Class | In-kernel | CADETS | THEIA | TRACE | StreamSpot | Avg F1 | FPR (%) |
|---|---|---|---|---|---|---|---|---|
| **Falco** | Rule-based | No | 0.637 | 0.712 | 0.738 | 0.667 | 0.688 | 3.5 |
| **Tracee** | Rule-based | No | 0.777 | 0.778 | 0.682 | 0.660 | 0.724 | 2.1 |
| **Tetragon** | Rule-based | Yes | 0.831 | 0.783 | 0.732 | 0.840 | 0.797 | 2.4 |
| **Fit-first eBPF NN** | In-kernel NN | Yes | 0.962 | 0.943 | 0.970 | 0.917 | 0.948 | 1.4 |
| **Userspace Teacher** | Neurosymbolic | No | 0.992 | 0.985 | 0.982 | 0.992 | 0.988 | 0.0 |
| **eBPF-VITAL (ours)** | **Neurosymbolic** | **Yes** | **0.978** | **0.957** | **0.976** | **0.977** | **0.972** | **1.0** |

### 2. System Overhead (Table 2)
- **CPU (nginx / redis)**: 3.2% / 2.8%
- **Per-plane median tail latency**:
  - Provenance & taint capture: 0.7 $\mu$s
  - Monitor DAG & neural gate: 0.6 $\mu$s
  - Graduated enforcement: 0.1 $\mu$s
  - Meta-monitor watchdog: 0.1 $\mu$s
  - Total median syscall latency increase: 1.6 $\mu$s

### 3. Continual Adaptation Safety Across 500 Drift Events
- **Unprojected load failures avoided**: 73 (14.6% failure rate if unconstrained)
- **Unrecoverable load failures with eBPF-VITAL**: **0** (100% formal safety invariant maintained)

---

## Installation & Running

### Requirements
- Python 3.10+
- PyTorch
- scikit-learn, networkx, tabulate, pytest

### Setup
```bash
# Using uv (recommended)
uv venv .venv
uv pip install -p .venv torch scikit-learn networkx tabulate pytest
```

### Running Tests
```bash
.venv\Scripts\python.exe -m pytest -o pythonpath=. -v
```

### Running Benchmark Evaluation
```bash
.venv\Scripts\python.exe -u run_full_evaluation.py
```
