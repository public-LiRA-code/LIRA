import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict

@dataclass
class Gate:
    name: str
    targets: Tuple[int, ...]
    params: Tuple[float, ...]

    def get_matrix(self, dtype=np.complex64) -> np.ndarray:
        n = self.name.lower()
        p = self.params

        if n == 'i':
            return np.eye(2, dtype=dtype)
        elif n == 'x':
            return np.array([[0, 1], [1, 0]], dtype=dtype)
        elif n == 'y':
            return np.array([[0, -1j], [1j, 0]], dtype=dtype)
        elif n == 'z':
            return np.array([[1, 0], [0, -1]], dtype=dtype)
        elif n == 'h':
            return np.array([[1, 1], [1, -1]], dtype=dtype) / np.sqrt(2)
        elif n == 's':
            return np.array([[1, 0], [0, 1j]], dtype=dtype)
        elif n == 'sdg':
            return np.array([[1, 0], [0, -1j]], dtype=dtype)
        elif n == 't':
            return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=dtype)
        elif n == 'tdg':
            return np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=dtype)
        elif n in ('p', 'u1'):
            theta = p[0] if p else 0.0
            return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=dtype)
        elif n == 'rz':
            theta = p[0] if p else 0.0
            return np.array([[np.exp(-1j * theta / 2), 0],
                             [0, np.exp(1j * theta / 2)]], dtype=dtype)
        elif n == 'rx':
            theta = p[0] if p else 0.0
            return np.array([[np.cos(theta/2), -1j * np.sin(theta/2)],
                             [-1j * np.sin(theta/2), np.cos(theta/2)]], dtype=dtype)
        elif n == 'ry':
            theta = p[0] if p else 0.0
            return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                             [np.sin(theta/2), np.cos(theta/2)]], dtype=dtype)
        elif n == 'u2':
            phi = p[0] if len(p) > 0 else 0.0
            lam = p[1] if len(p) > 1 else 0.0
            return np.array([[1, -np.exp(1j * lam)],
                             [np.exp(1j * phi), np.exp(1j * (phi + lam))]], dtype=dtype) / np.sqrt(2)
        elif n == 'u3' or n == 'u':
            theta = p[0] if len(p) > 0 else 0.0
            phi = p[1] if len(p) > 1 else 0.0
            lam = p[2] if len(p) > 2 else 0.0
            return np.array([
                [np.cos(theta/2), -np.exp(1j*lam) * np.sin(theta/2)],
                [np.exp(1j*phi) * np.sin(theta/2), np.exp(1j*(phi+lam)) * np.cos(theta/2)]
            ], dtype=dtype)
        elif n == 'cx' or n == 'cnot':
            return np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]
            ], dtype=dtype)
        elif n == 'cz':
            return np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, -1]
            ], dtype=dtype)
        elif n == 'cp':
            theta = p[0] if p else 0.0
            return np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, np.exp(1j * theta)]
            ], dtype=dtype)
        elif n == 'swap':
            return np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ], dtype=dtype)
        else:
            raise ValueError(f"Unsupported gate type: {n}")
