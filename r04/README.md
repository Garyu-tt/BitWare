# Mikoshi BitWeave R04 — one-P106 100 TOPS target (achieved conditionally)

**Status: experimental, validated on actual NVIDIA P106-100 Pascal SM 6.1, CUDA 12.4, GPU 0, 80 W configured power limit.** No changes to the other agent's DP4A project, existing GPU profiles, CUDA packages or production services.

## What 100 TOPS means here

We count **dense-equivalent binary arithmetic operations**, 2*M*N*K, over a mathematically exact sparse matrix multiplication. This is NOT 100 TFLOPS FP32, not 100 trillion physical integer operations, and not 100 TOPS on arbitrary dense matrices.

With 2048x2048 A, 2048x2048 B:
- B is block-sparse, each *group of 64 output channels* has independently sampled K-word positions (32 consecutive features per active word). Every group chooses different positions, so we are not merely deleting the same K columns for every output. All unselected weight positions are exactly 0. Selected weights are +/-1.
- GPU packs a binary (+/-1) A activation input into one sign bit per value using `__ballot_sync`.
- The kernel computes only nonzero blocks via XOR + POPCOUNT and produces EXACT results equivalent to dense INT8 cuBLAS GEMMEx, and independent CPU samples.
- B is prepacked in its persistent sparse representation once (one-time offline/transformation cost NOT timed).
- A is already GPU-resident as INT8 before packing (host-device transfer NOT timed).
- Tensor validation uses full dense cuBLAS INT32 reference. Output INT16 is guaranteed exact for SP<=32 (at most 1024 unit terms per output).

## Actual measured results, N=2048, K=2048

`sparse_gemm.cu` (INT32 output, kernel only):
- 12.5% active K-words (87.5% zero): 0.107304 ms = 160.10 dense-equivalent TOPS.
- 25% active (75% zero): 0.170917 ms = 100.52 dense-equivalent TOPS.
- 37.5% active: 0.248548 ms = 69.12 dense-equivalent TOPS.
- 50% active: 0.294719 ms = 58.29 dense-equivalent TOPS.

`sparse_gemm_pack.cu` (GPU packing benchmark, INT32 output):
- GPU input pack: 0.052204 ms, verified byte-for-byte vs CPU packing.
- 12.5% active kernel 0.111964 ms, total ~0.164168 ms = ~104.65 TOPS incl. pack, excluding PCIe.
- 25% active kernel 0.157522 ms = 109.06 TOPS kernel only, ~81.95 incl. GPU pack.

`sparse_gemm_pack_i16.cu` (GPU packing, INT16 exact output):
- First run: pack 0.052111 ms, 12.5% active kernel 0.099410 ms => ~113.43 dense-equivalent TOPS including GPU input pack.
- Repeats with 150 iterations:
  - 0.051808 + 0.091259 = 0.143067 ms => ~120.08 TOPS incl. pack.
  - 0.048729 + 0.092918 = 0.141647 ms => ~121.29 TOPS incl. pack.
  - 0.048715 + 0.092836 = 0.141551 ms => ~121.37 TOPS incl. pack.
- Exact output verification PASS over EVERY output versus cuBLAS GEMMEx INT32, plus 48 independent CPU samples, on every density tested.

### Important limitations
1. Above target requires HIGH structured sparsity and BINARY +/-1 activations. Arbitrary dense binary GEMM remains near the previous ~30 logical TOPS baseline. INT8 activations with ternary weights are a different kernel and are not benchmarked at 100+ TOPS here.
2. The full matrix-multiply-equivalent operation count includes positions that are skipped because the weights are zero. Useful dense-equivalent speedup is legitimate for this class of structured sparse workload, but do not compare it as if it were physical dense compute throughput.
3. No proof has been offered that a real LLM can tolerate removing 87.5% weights or 1-bit activations without significant model-quality loss. Sparse-aware training, calibration, and end-to-end benchmarks are required.
4. Prepacking B, host-device copies, output layer integration, and other neural-network stages are excluded. To match the measured ~121 TOPS, keep weights and activations on the GPU.
5. These numbers are microbenchmarks on a **single** P106-100, not multi-GPU totals and not tokens/second.

## Reproduce

```bash
cd BitWare
bash build.sh
CUDA_VISIBLE_DEVICES=0 ./build/sparse_gemm_pack_i16 --n 2048 --k 2048 --iters 150
```

Files: `sparse_gemm.cu` baseline, `sparse_gemm_pack.cu` GPU packing, `sparse_gemm_pack_i16.cu` half-size lossless output, `results_2048.csv`, `results_pack_2048.csv`, `results_i16_2048.csv`.

## Next genuine frontiers

- Autotune register tiles and packed intermediate format to reach 100 TOPS including dynamic pack at 75% (not 87.5%) structured sparsity.
- Train/evaluate a real binary-activation/structured block-sparse model; quantify quality, model size, and total token/s.
- Fuse preceding binary activation production with bitpacking to eliminate pack kernel entirely.
- Alternative non-sparsity techniques for dense arbitrary matrices need independent verification; do not rename a sparse win as dense throughput.
