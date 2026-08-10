# Grok-GPU-T4x2-Smoke — Kaggle run report

| Field | Value |
|---|---|
| Kernel | https://www.kaggle.com/code/xiaosuhuaer/grok-gpu-t4x2-smoke |
| Status | COMPLETE / SMOKE PASS |
| Accelerator | NvidiaTeslaT4 (2× Tesla T4, 15.64GB each) |
| DataParallel | true |
| Batch size | 256 |
| GEMM cuda:0 | ~4.17 TFLOPS FP32 |
| GEMM cuda:1 | ~4.42 TFLOPS FP32 |
| Train time | ~3.43 s (5 epochs, synthetic CNN) |
| Torch | 2.10.0+cu128 |
| Python | 3.12.13 |

Artifacts: `grok_gpu_t4x2_smoke_results.json`, `tiny_cnn_dp.pt` (not committed).
