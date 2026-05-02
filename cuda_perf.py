"""CUDA/cuBLAS/cuDNN tuning so Gemms use Tensor Core friendly paths where PyTorch supports them."""

from __future__ import annotations

import os


def configure_cuda_performance(*, log: bool = True) -> None:
    """Configure matrix-multiply and dnn backends before loading models.

    - FP32 matmuls: ``torch.set_float32_matmul_precision`` avoids the runtime warning on GPUs
      with Tensor Cores and enables TF32 *internal* math on Ampere+ when set to ``high`` or
      ``medium`` (see PyTorch docs). Volta has no TF32, but the precision setting still selects
      faster cuBLAS paths where available.
    - ``ECHO_CUDA_TF32``: enables TF32 for conv/matmul on **Ampere and newer** (no effect on
      Volta — harmless).
    - ``ECHO_CUDA_FP16_FAST_MATMUL``: FP16 GEMM accumulation path friendlier to Tensor Cores
      on GPUs like V100 when weights/activations are FP16.
    """
    import torch

    raw = (os.getenv("ECHO_MATMUL_PRECISION") or "high").strip().lower()
    if raw not in {"highest", "high", "medium"}:
        if log:
            print(f"[cuda] Unknown ECHO_MATMUL_PRECISION={raw!r}; using 'high'.", flush=True)
        raw = "high"
    torch.set_float32_matmul_precision(raw)

    tf32_on = (os.getenv("ECHO_CUDA_TF32", "1")).strip().lower() not in {"0", "false", "no", "off"}
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = tf32_on
        torch.backends.cudnn.allow_tf32 = tf32_on

    fp16_fast = (os.getenv("ECHO_CUDA_FP16_FAST_MATMUL", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = fp16_fast

    cudnn_benchmark = (os.getenv("ECHO_CUDNN_BENCHMARK", "0")).strip().lower() in {"1", "true", "yes", "on"}
    torch.backends.cudnn.benchmark = cudnn_benchmark

    if log:
        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability(0)
            print(
                f"[cuda] matmul_precision={raw} cuda_tf32={tf32_on} fp16_fast_matmul={fp16_fast} "
                f"cudnn_benchmark={cudnn_benchmark} device0_cc={cc[0]}.{cc[1]}",
                flush=True,
            )
        else:
            print(f"[cuda] matmul_precision={raw} (CPU — CUDA kernels inactive)", flush=True)
