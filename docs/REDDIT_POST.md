# We ran Qwen3-0.6B on the Strix Halo NPU at 4.8 tok/s — need help unlocking INT8

**TL;DR**: Reverse-engineered the undocumented AMD XDNA2 NPU on Strix Halo (Ryzen AI MAX+ 395) to run Qwen3-0.6B at **210ms/tok (4.8 tok/s) — 3.2× faster than CPU**. Built everything from scratch — 15 xclbins, 7 compiler bug fixes, full IRON API pipeline. But **INT8 is blocked by the MLIR toolchain** (only accepts BFP16 types), and **BF16 DMA hangs** due to aiecc descriptor generation bugs. Need community help.

---

## What We Built

```
                          ┌─────────────────────────────────────┐
                          │         Lemonade SDK / Ollama       │
                          │   (Single API — any model, any HW)  │
                          └──────────────┬──────────────────────┘
                                         │
                          ┌──────────────▼──────────────────────┐
                          │        ROCm HIP Runtime             │
                          └──────────────┬──────────────────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              │                          │                          │
    ┌─────────▼─────────┐    ┌──────────▼─────────┐    ┌──────────▼─────────┐
    │     GPU (GFX)     │    │     NPU (XDNA2)    │    │     CPU (x86)      │
    │  amdgpu driver    │    │  amdgpu NPU IP     │    │  Native (Zen 5)    │
    │  RDNA 3.5 CUs     │    │  8 AIE columns     │    │  ~2-3 tok/s CPU    │
    │  80 TFLOPS FP16   │    │  31 TFLOPS BFP16   │    │  (llama.cpp)       │
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

**The vision**: Fold `amdxdna` (NPU) into `amdgpu` so the NPU, GPU, and CPU share one memory manager, one DRM fd, and one ROCm compute API.

**The current stack (pre-unification)**:
```
6 custom xclbins → 4 GEMMs/layer × 28 layers = 112 NPU calls/token
Fused QKV (1024×4096) + Fused GU (1024×6144) + O + D projections
Threaded LM head (4×) + Threaded attention (4×)
BFP16 format (hardware-native block float, RMSE 0.0003)
```

## Performance

| Metric | CPU (llama.cpp) | NPU (This Work) | Gain |
|--------|----------------|------------------|------|
| Decode latency | ~668 ms/tok | **210 ms/tok** | **3.2×** |
| Throughput | ~1.5 tok/s | **4.8 tok/s** | **3.2×** |
| Power efficiency | ~25-35W | ~12W NPU+CPU | **~3× perf/W** |

## What Works

| Component | Status |
|-----------|--------|
| 6 BFP16 xclbins (QKV fused, O, GU fused, D, KV) | ✅ **Production** |
| 4 Multi-token M=256 xclbins (for batched decode) | ✅ Built |
| 4 2-layer batch N=8320 xclbins (for layer pairs) | ✅ Built |
| 4 INT8 xclbins (QKV, O, GU, D) | ✅ Built (DMA needs fix) |
| IRON API `@iron.jit` | ✅ Full pipeline, end-to-end |
| INT8 matmul via IRON (64×64×64) | ✅ Exact match, error=0 |

## 7 Compiler/API Bugs Fixed

All patched, tested, and documented:

1. **ScalarValue nanobind type mismatch** — removed ArithValueMeta metaclass
2. **AIE ELF symbol rename** — pure-Python ELF32 parser (objcopy doesn't handle 32-bit AIE ELFs)
3. **transpose.hpp incomplete type** — template deduction fallback
4. **mm.cc missing extern "C"** — Peano compiler needs C linkage
5. **SKIP_VECTORIZED flag** — preprocessor guard for scalar-only kernel
6. **MLIR parser i8/i16 rejection** — patched AIEXDialect.cpp + AIETargetModel.cpp
7. **~15 IRON API integration issues** — all fixed

## The Wall: INT8 and BF16

### INT8: 50 TOPS, Blocked by MLIR Dialect
```
✅ NPU hardware supports INT8 (proven by IRON API, error=0)
✅ MLIR parser patched to accept i8/i16
✅ INT8 xclbins built (66KB each)
❌ DMA strides need recalibration for 1-byte element types
   (vs BFP16's 1.125-byte packed format)
```

**Why**: The aiecc MLIR parser only accepts `v8bfp16ebs8`/`v16bfp16ebs16` types. We patched the source and rebuilt, but the DMA stride formulas in the MLIR generator were written for BFP16 and need adjustment for INT8's different element size.

### BF16: DMA Descriptor Bug
```
✅ BF16 xclbins compile
❌ Every variant hangs at runtime (identity, native, emulated)
❌ Chess compiler generates incorrect DMA descriptors for bfloat16
```

Windows handles BF16 correctly through a different NPU stack (DirectML/QNN).

## What We Need Help With

1. **INT8 DMA strides** — The n1_core tile streaming hierarchy in `n1_core_i8.py` produces DMA descriptors that hang for INT8. Someone familiar with AIE DMA multi-dimensional addressing (the `dimensionsToStream`/`dimensionsFromStream` attributes) could fix this in minutes.

2. **BF16 descriptor fix** — The Chess compiler's `bfloat16` DMA descriptor generation has a bug. Either a newer toolchain or a workaround in the MLIR generator.

3. **Windows NPU stack** — What does DirectML/QNN do differently? If someone can trace the Windows NPU driver calls for BF16/INT8, we can replicate it on Linux.

## Repository

Full handoff + xclbins + engine source + all investigation docs:

**https://github.com/bong-water-water-bong/npu-gpu-cpu/tree/main/docs**

| File | Content |
|------|---------|
| `HANDOFF-NPU-OPTIMIZATION.md` | Complete 3-day journey (880+ lines) |
| `INT8-HANDOFF.md` | INT8 deep-dive: 6 failed paths, root cause |
| `AGENTS.md` | Session summary |
| `REDDIT_POST.md` | This post |

---

*Built from scratch over 3 days. No documentation. No support. Just reverse engineering, 7 compiler bug fixes, 15 xclbins, and a lot of coffee.*
