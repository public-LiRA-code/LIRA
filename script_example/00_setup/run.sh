#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
bash build_engines.sh
source ../../setup_env.sh
"$LIRA_PYTHON" build_drivers.py
bash prepare_workloads.sh
