import os

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(CODE)
DATA = os.environ.get('LIRA_DATA', os.path.join(REPO, 'data'))
CIRCUITS = os.environ.get('LIRA_CIRCUITS', os.path.join(DATA, 'circuits'))
CASES = os.environ.get('LIRA_CASES', os.path.join(DATA, 'cases'))
RESULTS = os.environ.get('LIRA_RESULTS', os.path.join(REPO, 'results'))
THIRD_PARTY = os.environ.get('LIRA_THIRD_PARTY', os.path.join(REPO, 'third_party'))


def circuit(family: str, n: int) -> str:
    return os.path.join(CIRCUITS, f'MQTBench_{n}q', f'{family}_indep_qiskit_{n}_no_swap.qasm')
