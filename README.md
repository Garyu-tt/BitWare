# BitWare — BitWeave low-bit GPU research

Experimental CUDA binary/ternary bit-packed GEMM and GEMV, exact event-delta updates, and structured block-sparse kernels. **BitWare** is the repository name; **BitWeave** is the kernel architecture name.

**Status:** research microbenchmarks, not a production LLM inference engine, not an LLM checkpoint, and not a finished connectome simulator.

## Contents

- **r03/** — binary/ternary packed GEMM, signed INT8 activation plus binary/ternary weight GEMV, event updates, adaptive dispatch, tuning utilities, original results.
- **r04/** — structured block-sparse GEMM, GPU input packing, lossless INT16-output variant for supported sizes, original results.
- **build.sh** — build nine independent CUDA benchmark binaries.
- **scripts/verify_results.py** — offline arithmetic consistency checks for archived R04 measurements (not a GPU retest).

## Build and reproduce

Requires Linux, NVIDIA CUDA Toolkit (nvcc), cuBLAS, compatible C++ compiler, and a CUDA GPU. Originally benchmarked on **NVIDIA P106-100, Pascal sm_61, CUDA 12.4**, GPU configured with an 80 W power limit.

~~~sh
git clone https://github.com/Garyu-tt/BitWare.git
cd BitWare
bash build.sh
CUDA_VISIBLE_DEVICES=0 ./build/bitweave --n 2048 --iters 20 --ternary 0
CUDA_VISIBLE_DEVICES=0 ./build/bitweave_mixed --n 4096 --k 4096 --iters 80 --ternary 2
CUDA_VISIBLE_DEVICES=0 ./build/sparse_gemm_pack_i16 --n 2048 --k 2048 --iters 150
python3 scripts/verify_results.py
~~~

The build script targets sm_61 by default; set CUDA_ARCH and optionally CUDAHOSTCXX for other environments. Each program includes output checks against dense signed INT8 cuBLAS. See the R03 and R04 READMEs for experimental setup and caveats.

## Historical single-GPU benchmark results

| Workload | Reported timing | Interpretation |
| --- | --- | --- |
| R03 dense binary 2048² GEMM | 0.570 ms kernel | ~30.12 dense-equivalent TOPS; GPU input pack separately accounted |
| R03 signed INT8 activation + ternary weights, 4096-output GEMV | 0.04082 ms kernel + 0.00700 ms activation packing | 0.15731 ms cuBLAS reference |
| R04 87.5% sparse binary 2048² GEMM, INT32 output | 0.107304 ms kernel | 160.10 dense-equivalent TOPS, input pack excluded |
| R04 87.5% sparse binary 2048² GEMM, INT16 output | 0.091259 ms kernel + 0.051808 ms input pack | ~120.08 dense-equivalent TOPS including separately timed input packing |

**Dense-equivalent TOPS = 2*M*N*K / elapsed seconds**, including computations skipped because sparse weights are zero. It is **not** physical integer or floating-point compute throughput, not comparable to silicon FLOPS, and not LLM tokens/second. High results require special structured sparsity and binary activations; the effect on real-model accuracy is not established.

These are archived experiment results, **not independent GPU retests performed for publication**. A later reported **225.67 dense-equivalent TOPS** result is not included as reproducible in this snapshot because the corresponding source and measurements were not found in the inspected available directories.

## Correctness and limits

- Verified results compare to dense cuBLAS on tested synthetic input shapes and include some independent CPU samples. This does not establish real LLM accuracy after aggressive quantization/pruning.
- Microbenchmarks exclude static-weight prepacking, host/device transfer, attention, KV cache, scheduler, full-model inference, and inter-GPU communication.
- Sparse event benchmarks assume GPU-resident events and output state; event detection/compaction is excluded.
- 80 W is the configured device power limit, **not** a published continuous power-consumption trace.
- Reproducing results requires a compatible NVIDIA GPU. The CSV audit script checks arithmetic consistency only.

## License and related work

GNU General Public License v3 (see LICENSE), matching the related [Mikoshi HQ500M](https://github.com/Garyu-tt/mikoshi-hq500m) public project. BitWare is standalone and does not require Noosphere or HQ500M to compile these programs.
