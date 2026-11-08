#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
NVCC="$(printenv NVCC || true)"
ARCH="$(printenv CUDA_ARCH || true)"
HOSTCXX="$(printenv CUDAHOSTCXX || true)"
if [[ -z "$NVCC" ]]; then NVCC=nvcc; fi
if [[ -z "$ARCH" ]]; then ARCH=sm_61; fi
compiler_args=()
if [[ -n "$HOSTCXX" ]]; then
  compiler_args=(-ccbin "$HOSTCXX")
elif command -v g++-13 >/dev/null 2>&1; then
  compiler_args=(-ccbin "$(command -v g++-13)")
fi
mkdir -p build
for n in bitweave bitweave_tune bitweave_gemv bitweave_mixed bitweave_events bitweave_adaptive; do
  "$NVCC" -O3 -std=c++17 "-arch=$ARCH" "${compiler_args[@]}" "r03/$n.cu" -lcublas -o "build/$n"
done
for n in sparse_gemm sparse_gemm_pack sparse_gemm_pack_i16; do
  "$NVCC" -O3 -std=c++17 "-arch=$ARCH" "${compiler_args[@]}" "r04/$n.cu" -lcublas -o "build/$n"
done
echo "BitWare R03/R04 binaries built in ./build/"
