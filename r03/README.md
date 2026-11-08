# Mikoshi BitWeave — R03 experimental low-bit / event-driven CUDA backend

Research-only standalone sibling of the DP4A GEMM project. No production services, drivers, GPU limits, system CUDA packages, LLM checkpoints, or existing repositories were modified.

**Test device:** NVIDIA P106-100, Pascal sm_61, CUDA 12.4, `nvcc -ccbin /usr/bin/g++-13`; GPU 0, configured 80 W power limit. Results are measured CUDA-event kernel latencies, not whole-LLM token/s, across single GPU.

## Versions

- `bitweave.cu`: exact binary +/-1 and ternary {-1,0,1} packed GEMM for **both** operands. GPU-side `__ballot_sync` packing; tiled XNOR+POPCOUNT or masked ternary POPCOUNT. Baseline signed INT8 cuBLAS GEMMEx and verification of **every** output.
- `bitweave_tune.cu`: tiled binary kernel autotuning of 32×64, 64×64, 32×128, 64×128 tiles and bitwise Ktiles=8/16 packed words. Larger tiles did not improve over 32×64 in tested sizes.
- `bitweave_gemv.cu`: one-token binary activation with binary or ternary weights, warp-per-output and shuffle reduction.
- `bitweave_mixed.cu`: **signed INT8 activation** (8 two's-complement bit planes) with binary or ternary packed weights. Exact INT32 output relative to cuBLAS INT8, all outputs verified, with 32 independent CPU outputs checked. This is materially closer to ternary-weight LLMs.
- `bitweave_events.cu`: INT8 activation updates with sparse event list; exact `y_new = y_old + W*(x_new-x_old)`, ternary W, all outputs checked vs full cuBLAS.
- `bitweave_adaptive.cu`: benchmarks and dispatches between event-delta update and full bitplane GEMV (including dynamic activation packing), verifies both paths and the chosen result. Research selector currently calibrates from on-run timings; production should select from offline calibrated thresholds, not measure every inference.

## Build

```bash
cd BitWare
bash build.sh
CUDA_VISIBLE_DEVICES=0 ./build/bitweave --n 2048 --iters 20 --ternary 0
CUDA_VISIBLE_DEVICES=0 ./build/bitweave_mixed --n 4096 --k 4096 --iters 80 --ternary 2
CUDA_VISIBLE_DEVICES=0 ./build/bitweave_adaptive --n 4096 --k 4096 --iters 70
```

`--ternary 0` = binary both, `--ternary 1` = ternary both (GEMM) or ternary weights with binary activations (GEMV).
`--ternary 2` = ternary weights and signed INT8 activations, `--ternary 3` = binary weights and signed INT8 activations (mixed).
Commands default to GPU 0 and should be run when it is free.

## Observed single-GPU numbers

**GEMM: square 2048×2048, prepacked matrices**
- Binary +/-1 x +/-1: 0.570 ms, ~30.12 equivalent TOPS. cuBLAS INT8 1.172 ms (2.05× kernel speedup; 1.69× including one input pack).
- Ternary x ternary: 1.225 ms vs cuBLAS INT8 1.244 ms; packing removes win. No claim of general superiority.

**GEMV: one vector, 4096 outputs, K=4096**
- Binary weights and binary activation: 0.01782 ms vs cuBLAS INT8 0.16645 ms; 6.69× including dynamic vector pack.
- Ternary weights and binary activation: 0.02778 ms vs 0.15663 ms; 4.50× including dynamic pack.
- Ternary weights with **signed INT8 activation**: 0.04082 ms vs cuBLAS INT8 0.15731 ms; 3.29× including 0.00700 ms activation pack.
- Binary weights with **signed INT8 activation**: 0.02998 ms vs cuBLAS INT8 0.15686 ms; 4.24× including 0.00701 ms pack.
- Prepacking weights is a **one-time cost** (0.5–1 ms at this geometry), not included in repeated GEMV latency; host-to-device copies not included.

**Adaptive: ternary weights, signed INT8 activation, previous layer output GPU-resident**
`adaptive_r01.csv` contains direct same-run full BitWeave and event delta comparisons, and the verified chosen path.
- 1 event -> event delta ~0.010–0.011 ms vs full mixed BitWeave ~0.046–0.048 ms (~4.5× faster).
- 64 events -> event delta ~0.031 ms vs full ~0.046–0.048 ms (~1.5×).
- 256 events -> event ~0.040 ms vs full ~0.045 ms: small gain.
- 384+ events -> full BitWeave wins. Conservative dispatcher threshold could be **<=64 events** at this tested shape until event detection and transfer cost are known; full benchmarking is size and device dependent.
- Events are already provided as device-resident indices+deltas when timed. Event detection, compaction, uploads, and past-output initialization are **not** included; these are part of production costs.

## Correctness, practical limitations

1. All measured binary, ternary, mixed, and event outputs matched full INT8 cuBLAS exactly on tested inputs, plus independent CPU samples. Synthetic weights are chosen {-1,0,1} or {-1,1}; this does **not** establish output quality if weights from a conventional LLM are abruptly quantized. Real low-bit networks need training, quantization-aware calibration or model designed for ternary weights.
2. The binary-only path requires +/-1 activations and is not equivalent to general INT8 activation inference.
3. Matrix multiplication benchmark includes only GPU execution and separately reports input packing; final token/s also depends on attention, KV-cache, routing, CPU-GPU transfers, scheduling, quantization scales, and other model layers.
4. Event-driven update is most useful for sparse, slowly changing state (connectome-like neural state, recurrent systems, sensor streams), **not** automatically for a transformer whose every activation changes every token.
5. Persistent prepacked weights on each GPU are important: the eight P106-100 adapters use PCIe x1 risers, making frequent host/device or cross-device tensor transfer expensive. Do not assume 8× speedup across GPUs.
6. Report logical equivalent operations (2*M*N*K / time) for bitwise GEMM, **not** literal hardware float FLOPS or measured full-LLM performance.

## Next performance directions

- **R04** real LLM layer integration: calibrate ternary weights/INT8 activations per-channel and report perplexity/quality + per-layer speed; add zero-points/scales, fused bias/activation.
- **R05** CUDA kernel optimization: event compaction on GPU, small-event specialized kernels, occupancy autotuner, persistent GPU stream/graph, mixed 4/8-bit bitplanes.
- **R06** energy-aware scheduler: route dense vs sparse vs CPU/GPU based on event density, batch size, memory locality and actual watts. Cross-GPU layers should be coarse and weight-stationary to avoid PCIe x1.
