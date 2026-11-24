#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source setup_env.sh
(cd "$LIRA_CIRCUITS" && sha256sum -c --quiet SHA256SUMS) && echo "circuits: checksums match"
"$LIRA_PYTHON" lira_code/benchmarks/export_case.py --n 24,25,26,27,28,29,30,31 --cases sel_4 --obs meanz --out "$LIRA_CASES"
