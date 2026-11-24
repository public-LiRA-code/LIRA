#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source setup_env.sh
mkdir -p "$LIRA_THIRD_PARTY" "$LIRA_ENVS"
TP=$LIRA_THIRD_PARTY
PATCH=$LIRA_HOME/lira_code/patches/lightning-kokkos-v0.45.0-receive-rank.patch

clone() {
    [ -d "$TP/$1" ] || git clone -q "$2" "$TP/$1"
    git -C "$TP/$1" checkout -q "$3"
}
clone pennylane-lightning https://github.com/PennyLaneAI/pennylane-lightning.git 862df1a4e855d1519ae5ec5ed2df3688268170d4
clone kokkos https://github.com/kokkos/kokkos.git 3ec81abe1816109f6f62ac48cef41921f91a4d00
clone qulacs https://github.com/qulacs/qulacs.git 6fbf01ca509754705134d9a65fdcca4a1c167500
clone torchquantum https://github.com/mit-han-lab/torchquantum.git 8dc3255c51477dd4c28892049571df032c77e2ff
clone VQCSim https://github.com/Security-FIT/VQCSim.git 7fc274d448c5e67a7a4149088afa594203cd5707
clone triton_qml https://github.com/puyokw/triton_qml.git f32bb78e3f95ff744cfcd91b0a6ff7d63c5c19b0

[ -d "$LIRA_ENVS/lira" ] || python -m venv "$LIRA_ENVS/lira"
"$LIRA_ENVS/lira/bin/python" -m pip install -q -r requirements.txt
MPICC="cc -shared" "$LIRA_ENVS/lira/bin/python" -m pip install -q --force-reinstall --no-binary=mpi4py mpi4py==4.1.2
(
    source "$LIRA_ENVS/lira/bin/activate"
    cd "$TP/pennylane-lightning"
    rm -rf build build_lightning_gpu
    CUQUANTUM_SDK=$(python -c "import site; print(site.getsitepackages()[0])")/cuquantum
    PL_BACKEND=lightning_gpu python scripts/configure_pyproject_toml.py
    CC=cc CXX=CC CMAKE_BUILD_PARALLEL_LEVEL=8 \
        CMAKE_ARGS="-DENABLE_MPI=ON -DCUQUANTUM_SDK=$CUQUANTUM_SDK -DCMAKE_CUDA_ARCHITECTURES=80" \
        python -m pip install . --no-deps -q
    git checkout -q -- .
)

KINST=$TP/kokkos_install
if [ ! -d "$KINST" ]; then
    cmake -S "$TP/kokkos" -B "$TP/kokkos/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$KINST" \
        -DCMAKE_CXX_COMPILER=CC -DCMAKE_CXX_STANDARD=20 -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON \
        -DKokkos_ENABLE_COMPLEX_ALIGN=OFF -DKokkos_ENABLE_EXAMPLES=OFF -DKokkos_ENABLE_TESTS=OFF -DKokkos_ENABLE_LIBDL=OFF
    cmake --build "$TP/kokkos/build" -j 16
    cmake --install "$TP/kokkos/build"
fi
[ -d "$LIRA_ENVS/lightning-kokkos" ] || python -m venv "$LIRA_ENVS/lightning-kokkos"
"$LIRA_ENVS/lightning-kokkos/bin/python" -m pip install -q "pennylane==0.45.1" scipy-openblas32 cmake ninja
MPICC="cc -shared" "$LIRA_ENVS/lightning-kokkos/bin/python" -m pip install -q --force-reinstall --no-binary=mpi4py mpi4py==4.1.2
(
    source "$LIRA_ENVS/lightning-kokkos/bin/activate"
    rm -rf "$TP/pennylane-lightning-kokkos"
    git clone -q --shared "$TP/pennylane-lightning" "$TP/pennylane-lightning-kokkos"
    cd "$TP/pennylane-lightning-kokkos"
    git checkout -q 862df1a4e855d1519ae5ec5ed2df3688268170d4
    git apply "$PATCH"
    PL_BACKEND=lightning_qubit python scripts/configure_pyproject_toml.py
    SKIP_COMPILATION=True python -m pip install . --no-deps -q
    PL_BACKEND=lightning_kokkos python scripts/configure_pyproject_toml.py
    CMAKE_PREFIX_PATH="$KINST" CC=cc CXX=CC CMAKE_BUILD_PARALLEL_LEVEL=16 \
        MPI_EXTRA_LINKER_FLAGS="${PE_MPICH_GTL_DIR_nvidia80:-} ${PE_MPICH_GTL_LIBS_nvidia80:-}" \
        CMAKE_ARGS="-DENABLE_MPI=ON -DCMAKE_CXX_COMPILER=CC" python -m pip install . --no-deps -q
)

(
    source "$LIRA_ENVS/lira/bin/activate"
    USE_GPU=Yes C_COMPILER=gcc CXX_COMPILER=g++ python -m pip install -q "$TP/qulacs" --no-deps --target "$LIRA_QULACS_PATH"
)

TORCH_BASE=${LIRA_TORCH_BASE:-python}
[ -d "$LIRA_ENVS/torch" ] || "$TORCH_BASE" -m venv --system-site-packages "$LIRA_ENVS/torch"
"$LIRA_ENVS/torch/bin/python" -m pip install -q -e "$TP/torchquantum" "qiskit==2.5.2"

"$LIRA_JULIA" --project="$LIRA_HOME/lira_code/benchmarks/yao_env" -e 'using Pkg; Pkg.instantiate()'
