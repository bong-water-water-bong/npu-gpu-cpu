# 🚀 NPU + GPU + CPU = Unified Control Plane

**Single kernel driver + unified memory for AMD Strix Halo on Linux.**

Folding `amdxdna` (XDNA 2 NPU) into `amdgpu` so the NPU, GPU, and CPU share one memory manager, one DRM file descriptor, and one ROCm compute API.

---

## 🔥 BREAKTHROUGH: Qwen3-0.6B Running on the NPU at 4.8 tok/s

**We got a production LLM running on the Strix Halo NPU — 210ms/tok, 3.2× faster than baseline.**

This is **real inference on real hardware** — 28-layer transformer, BFP16 precision, 6 custom xclbins, threaded CPU pipeline. All on the Linux XRT stack with the torch2aie/IRON toolchain. No Windows. No proprietary runtime. Just raw NPU compute.

### The Stack
| Layer | Tech |
|-------|------|
| **Model** | Qwen2.5-0.6B (600M params, 28 layers) |
| **NPU** | Strix Halo XDNA2 — 8 AIE columns, 31 TFLOPS BFP16 peak |
| **Precision** | BFP16 (v8bfp16ebs8) — hardware-native block float, RMSE 0.0003 |
| **Toolchain** | MLIR-AIE + aiecc + Chess compiler |
| **Runtime** | XRT (Xilinx Runtime) + custom C++ engine |
| **GEMMs** | 112 NPU calls/token — fused QKV, GU, O, D projections |

### Performance
| Metric | Before | After | Gain |
|--------|--------|-------|------|
| Decode latency | 668 ms/tok | **210 ms/tok** | **3.2×** |
| Throughput | 1.5 tok/s | **4.8 tok/s** | **3.2×** |
| Prefill (9 tok) | — | 1.67s | — |
| Init (cached) | — | 2.5s | — |

### 15+ Built XCLBIN Artifacts
| Type | Count | Status |
|------|-------|--------|
| BFP16 projection xclbins (QKV, O, GU, D) | 6 | ✅ **Running in production** |
| Multi-token M=256 xclbins | 4 | ✅ Built (needs aiecc instruction fix) |
| 2-layer batch N=8320 xclbins | 4 | ✅ Built (needs engine integration) |
| INT8 xclbins | 4 | ✅ Built (needs DMA stride fix) |

### 🔧 7 Compiler/API Bugs Fixed
All fixed and upstream-ready — MLIR bindings, ELF rename, transpose template, kernel extern "C", vectorized/skipal flags. IRON API `@iron.jit` works end-to-end.

---

## ⛔ The Walls We Hit

### INT8 — 50 TOPS, Blocked by Software
The hardware supports INT8 natively (we proved it with IRON API — exact match, error=0). But the MLIR parser **only accepts `v8bfp16ebs8`/`v16bfp16ebs16` types** — `i8` is rejected. We patched the aiecc source, rebuilt, and built working INT8 xclbins (66KB). The last mile: recalibrating DMA strides for 1-byte element types. **This is a software limitation, not hardware.**

### BF16 — Hangs at Runtime
BF16 xclbins compile but the DMA controller hangs. Every kernel variant (identity, native, emulated) hangs identically. The Chess compiler generates incorrect DMA descriptors for `bfloat16` memory types. Windows handles this correctly through a different NPU stack.

---

## 📂 Project Structure

```
├── AGENTS.md                          # Project standards
├── openspec/changes/
│   └── unified-amdgpu-driver/
│       ├── proposal.md                # Architecture proposal
│       └── tasks.md                   # Implementation tracking
├── docs/
│   ├── AGENTS.md                      # NPU session summary
│   ├── HANDOFF-NPU-OPTIMIZATION.md    # Complete optimization journey (880+ lines)
│   ├── INT8-HANDOFF.md                # INT8 deep-dive (~6.5KB)
│   ├── REDDIT_POST.md                 # Community call for help (draft)
│   └── wiki/                          # Durable project knowledge
├── patches/                           # Kernel patches
├── rocm-npu/                          # ROCm runtime NPU target
│   ├── hip_npu.cpp
│   └── CMakeLists.txt
└── daemon/                            # npu-gpu-cpud scheduler
```

## Layers

| Layer | What | Where |
|-------|------|-------|
| Kernel | `amdgpu` NPU IP block | `drivers/gpu/drm/amd/amdgpu/` |
| Runtime | ROCm HIP NPU agent | `rocm-npu/` (this repo) |
| Scheduler | `npu-gpu-cpud` daemon | `daemon/` (this repo) |

## Status

**Phase 0: Planning** — NPU inference proven. Kernel module proposal written. INT8/BF16 blocked by toolchain — community help needed.

---

*Built from scratch over 3 days. No documentation. No support. Just reverse engineering, 7 compiler bug fixes, 15 xclbins, and a lot of coffee.*
