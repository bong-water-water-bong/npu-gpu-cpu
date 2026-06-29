# 🚀 NPU + GPU + CPU = Unified Control Plane

**Single kernel driver + unified memory for AMD Strix Halo on Linux.**

Folding `amdxdna` (XDNA 2 NPU) into `amdgpu` so the NPU, GPU, and CPU share one memory manager, one DRM file descriptor, and one ROCm compute API.

---

## The Vision: One Ring to Rule Them All

```
                          ┌─────────────────────────────────────┐
                          │         Lemonade SDK / Ollama       │
                          │   (Single API — any model, any HW)  │
                          └──────────────┬──────────────────────┘
                                         │
                          ┌──────────────▼──────────────────────┐
                          │        ROCm HIP Runtime             │
                          │  (hipMalloc, hipMemcpy, hipLaunch)  │
                          └──────────────┬──────────────────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              │                          │                          │
    ┌─────────▼─────────┐    ┌──────────▼─────────┐    ┌──────────▼─────────┐
    │     GPU (GFX)     │    │     NPU (XDNA2)    │    │     CPU (x86)      │
    │  amdgpu driver    │    │  amdgpu NPU IP     │    │  amdgpu ✗ (native) │
    │  RDNA 3.5 CUs     │    │  8 AIE columns     │    │  AMD Zen 5 cores   │
    │  50 TOPS INT8     │    │  31 TFLOPS BFP16   │    │  ~2-3 tok/s CPU    │
    │  80 TFLOPS FP16   │    │  50 TOPS INT8      │    │  (llama.cpp)       │
    └─────────┬─────────┘    └──────────┬─────────┘    └──────────┬─────────┘
              │                          │                          │
              └──────────────────────────┼──────────────────────────┘
                                         │
                          ┌──────────────▼──────────────────────┐
                          │       Unified Memory Manager        │
                          │   One DRM fd, one address space     │
                          │   Shared page table (GPU→NPU→CPU)   │
                          └─────────────────────────────────────┘
```

**The problem today**: Three separate drivers. Three memory spaces. Three programming models. The NPU uses XRT + xclbins. The GPU uses ROCm + HIP. The CPU uses llama.cpp. None of them talk to each other.

**The goal**: One `amdgpu` driver that sees the NPU as just another compute engine. One `hipMalloc` that works across all three. One runtime API that schedules work on whatever hardware is available.

---

## 🔥 BREAKTHROUGH: Qwen3-0.6B Running on the NPU at 4.8 tok/s

**We got a production LLM running on the Strix Halo NPU — 210ms/tok, 3.2× faster than CPU baseline.**

This is **real inference on real hardware** — 28-layer transformer, BFP16 precision, 6 custom xclbins, threaded CPU pipeline. All on the Linux XRT stack with the torch2aie/IRON toolchain. No Windows. No proprietary runtime. Just raw NPU compute.

### The Current Stack (Pre-Unification)
```
┌─────────────────────────────────────────────────────────────┐
│  Custom C++ Engine (npu_infer_fused)                        │
│  ├── XRT API → xclbin BOs → AIE DMA → compute tiles        │
│  ├── Thread pool: 4× LM head, 4× attention                 │
│  └── Weight cache: Q4NX → dequant → BFP16 pack → NPU DMA   │
├─────────────────────────────────────────────────────────────┤
│  torch2aie / IRON toolchain                                 │
│  ├── MLIR-AIE → aiecc → Chess kernel .o → xclbin           │
│  └── @iron.jit JIT compilation pipeline (fully working)     │
├─────────────────────────────────────────────────────────────┤
│  Linux XRT + amdxdna (standalone NPU driver)                │
│  └── DRM ioctl → HW context → BO create/sync → exec cmd    │
└─────────────────────────────────────────────────────────────┘
```

### Performance
| Metric | CPU (llama.cpp) | NPU (This Work) | Gain |
|--------|----------------|------------------|------|
| Decode latency | ~668 ms/tok | **210 ms/tok** | **3.2×** |
| Throughput | ~1.5 tok/s | **4.8 tok/s** | **3.2×** |
| Prefill (9 tok) | — | 1.67s | — |
| Power | ~25-35W (Zen 5) | ~2W NPU + ~10W CPU | **~3× perf/W** |

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

### INT8 — 50 TOPS, Blocked by Software, Not Hardware
The NPU supports INT8 natively. **We proved it**: IRON API INT8 matmul at 64×64×64 produces **exact match, error=0**. But the MLIR parser in aiecc **only accepts `v8bfp16ebs8`/`v16bfp16ebs16` types** — `i8` is rejected at parse time.

**What we did**: Patched `AIEXDialect.cpp` and `AIETargetModel.cpp` in the aiecc source, rebuilt with ninja, and successfully built working INT8 xclbins (66KB). The last mile is recalibrating DMA strides for 1-byte element types (vs BFP16's 1.125-byte packed format).

**Why it matters**: INT8 unlocks **2× throughput** (~100ms/tok, ~10 tok/s) and is the standard for quantized LLM inference.

### BF16 — Hangs at Runtime
BF16 xclbins compile but the DMA controller hangs. Every kernel variant (identity, native, emulated) hangs identically. The Chess compiler generates incorrect DMA descriptors for `bfloat16` memory types. Windows handles this correctly through a different NPU stack (DirectML/QNN).

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
├── patches/                           # Kernel patches against linux.git
├── rocm-npu/                          # ROCm runtime NPU target
│   ├── hip_npu.cpp
│   └── CMakeLists.txt
└── daemon/                            # npu-gpu-cpud scheduler
```

### Layered Architecture

```
┌──────────────────────────────────────────────────────────┐
│  USERSPACE                                                │
│  ┌─────────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Lemonade SDK   │  │  Ollama      │  │  Custom App  │ │
│  │  (OpenAI API)   │  │  (llama.cpp) │  │  (C++/Python)│ │
│  └────────┬────────┘  └──────┬───────┘  └──────┬───────┘ │
│           │                  │                  │          │
│  ┌────────▼──────────────────▼──────────────────▼───────┐ │
│  │              ROCm HIP Runtime (Unified)              │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │ │
│  │  │ hip_amg  │  │ hip_npu  │  │ hip_cpu (fallback)│   │ │
│  │  │ (GPU)    │  │ (NPU)    │  │ (CPU)             │   │ │
│  │  └─────┬────┘  └────┬─────┘  └────────┬─────────┘   │ │
│  └────────┼─────────────┼─────────────────┼──────────────┘ │
├───────────┼─────────────┼─────────────────┼────────────────┤
│  KERNEL   │             │                 │                │
│  ┌────────▼─────────────▼─────────────────▼──────────────┐ │
│  │              amdgpu (Unified Driver)                  │ │
│  │  ┌──────────┐  ┌──────────────┐  ┌────────────────┐  │ │
│  │  │ GFX IP   │  │ NPU IP       │  │  DMA-BUF       │  │ │
│  │  │ (gfx1151)│  │ (xDNA2 AIE)  │  │  Shared PT     │  │ │
│  │  └──────────┘  └──────────────┘  └────────────────┘  │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

## Status

**Phase 0: Planning** — NPU inference proven at 210ms/tok. Kernel module proposal written. INT8/BF16 blocked by toolchain — community help needed.

---

*Built from scratch over 3 days. No documentation. No support. Just reverse engineering, 7 compiler bug fixes, 15 xclbins, and a lot of coffee.*

**[github.com/bong-water-water-bong/npu-gpu-cpu](https://github.com/bong-water-water-bong/npu-gpu-cpu)**
