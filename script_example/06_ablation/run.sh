#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source setup_env.sh
"$LIRA_PYTHON" script_example/common/run.py 06_ablation "$@"
