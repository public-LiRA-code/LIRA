#!/bin/bash
module load python
module load cudatoolkit/12.9
module load cray-mpich/9.0.1
module load craype-accel-nvidia80
module unload cray-libsci

export LIRA_HOME=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export LIRA_DATA=${LIRA_DATA:-$LIRA_HOME/data}
export LIRA_CIRCUITS=${LIRA_CIRCUITS:-$LIRA_DATA/circuits}
export LIRA_CASES=${LIRA_CASES:-$LIRA_DATA/cases}
export LIRA_RESULTS=${LIRA_RESULTS:-$LIRA_HOME/results}
export LIRA_THIRD_PARTY=${LIRA_THIRD_PARTY:-$LIRA_HOME/third_party}
export LIRA_ENVS=${LIRA_ENVS:-$LIRA_HOME/envs}
export LIRA_PYTHON=${LIRA_PYTHON:-$LIRA_ENVS/lira/bin/python}
export LIRA_LKK_PYTHON=${LIRA_LKK_PYTHON:-$LIRA_ENVS/lightning-kokkos/bin/python}
export LIRA_TORCH_PYTHON=${LIRA_TORCH_PYTHON:-$LIRA_ENVS/torch/bin/python}
export LIRA_QULACS_PATH=${LIRA_QULACS_PATH:-$LIRA_THIRD_PARTY/qulacs_pkg}
export LIRA_JULIA=${LIRA_JULIA:-julia}
export JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-$LIRA_ENVS/julia_depot}
export LIRA_MPIPROF_LIB=$LIRA_HOME/lira_code/tools/libmpiprof.so
export LIRA_GPUS_PER_NODE=${LIRA_GPUS_PER_NODE:-4}

export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU
