# SLM Latency Analysis & Optimization Report

**Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)
**Model:** microsoft/Phi-3-mini-4k-instruct (4-bit Quantized via BitsAndBytes)
**Framework:** Transformers Pipeline

## 1. Benchmark Results

We conducted a systematic benchmark to isolate the latency bottlenecks.

| Benchmark Configuration | Avg Generation Time | Token Count (Input) | Notes |
| :--- | :--- | :--- | :--- |
| **Baseline** (Robust Prompt) | ~1978 ms | ~505 tokens | `max_new_tokens=15` |
| **Optimized** (Compact Prompt) | ~1419 ms | ~100 tokens | `max_new_tokens=15` |

### Key Findings
1.  **Prefill is significant:** Reducing the prompt from ~505 tokens to ~100 tokens reduced total latency by **~28% (560ms)**. This confirms that prompt processing (prefill) is a major component of the latency on this hardware.
2.  **Generation is still slow:** Even with a tiny prompt (~100 tokens) and minimal generation (15 tokens), the total time is **~1.4s**. This suggests the `transformers` pipeline overhead and `bitsandbytes` 4-bit dequantization on-the-fly are the primary bottlenecks.
3.  **Timeout Limit:** The current 500ms timeout for the hybrid fast-path is consistently exceeded. The SLM path effectively never "wins" the race against the timeout.

## 2. Root Cause Analysis

Why is it ~1.4s even with optimization?
*   **Transformers Overhead:** The Hugging Face `pipeline` is optimized for flexibility, not raw inference latency. It has significant Python overhead per call.
*   **BitsAndBytes 4-bit:** While memory efficient, `bitsandbytes` quantization can be slower than FP16 inference because weights must be dequantized before computation.
*   **Eager Attention:** We are using `attn_implementation="eager"` to avoid `DynamicCache` errors, which forfeits the speedups from Flash Attention 2.

## 3. Recommendation: Move to Phase 3 (GGUF/Llama.cpp)

To get latency under **500ms** (ideally <200ms) on an RTX 4060, we need a highly optimized inference backend.

**Proposed Next Steps:**
1.  **Install `llama-cpp-python` with CUDA support.** This backend uses the GGUF format, which is purpose-built for fast inference on consumer hardware.
2.  **Switch Model Format:** Download the GGUF version of Phi-3-Mini (e.g., `Phi-3-mini-4k-instruct-q4_k_m.gguf`).
3.  **Benchmark GGUF:** We expect this to reduce latency to the **100-300ms** range, making the local SLM viable for the fast-path.

Does this align with your goals?

