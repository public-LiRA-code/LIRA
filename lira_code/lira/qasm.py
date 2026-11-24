import re
import numpy as np
from typing import Tuple, List
from .gates import Gate

def parse_qasm(filepath: str) -> Tuple[int, List[Gate]]:
    n_qubits = 0
    gates = []

    qreg_offsets = {}
    current_offset = 0
    n_qubits = 0

    qreg_pattern = re.compile(r"qreg\s+([a-zA-Z_]\w*)\[(\d+)\];")

    gate_pattern = re.compile(r"^([a-z0-9_]+)(?:\(([^)]+)\))?\s+(.+);")

    target_pattern = re.compile(r"([a-zA-Z_]\w*)\[(\d+)\]")

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("OPENQASM") or line.startswith("include"):
                continue

            m_qreg = qreg_pattern.match(line)
            if m_qreg:
                name = m_qreg.group(1)
                size = int(m_qreg.group(2))
                qreg_offsets[name] = current_offset
                current_offset += size
                n_qubits = current_offset
                continue

            m_gate = gate_pattern.match(line)
            if m_gate:
                gate_name = m_gate.group(1).lower()
                params_str = m_gate.group(2)
                targets_str = m_gate.group(3)

                if gate_name in ('barrier', 'measure', 'reset', 'creg', 'opaque'):
                    continue

                params = ()
                if params_str:
                    try:
                        eval_ns = {"pi": np.pi, "np": np}
                        parts = params_str.split(',')
                        params = tuple(float(eval(p.strip(), {"__builtins__": None}, eval_ns)) for p in parts)
                    except Exception as e:
                        print(f"Warning: could not parse params '{params_str}' in line: {line}. Error: {e}")
                        params = tuple()

                targets = []
                for q_name, q_idx in target_pattern.findall(targets_str):
                    if q_name in qreg_offsets:
                        targets.append(qreg_offsets[q_name] + int(q_idx))
                    else:
                        print(f"Warning: Unknown qreg '{q_name}' in line: {line}")

                if targets:
                    gates.append(Gate(name=gate_name, targets=tuple(targets), params=params))

    return n_qubits, gates
