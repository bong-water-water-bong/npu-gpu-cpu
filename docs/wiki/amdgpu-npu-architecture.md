# AMD NPU+GPU+CPU Unified Control Plane — Architecture

## Hardware

**AMD Strix Halo (Ryzen AI MAX+ 395)**

| Die | Arch | Compute | Memory | Power | Linux Driver |
|-----|------|---------|--------|-------|-------------|
| **CPU** | Zen 5, 32 cores | ~1 TFLOPS | DDR5 (shared) | 15-35W | `core` |
| **GPU** | RDNA 3.5, gfx1151, 20 CUs | ~12 TFLOPS | DDR5 (shared, 64 GB) | 15-25W | `amdgpu` |
| **NPU** | XDNA 2, aie2, 8 columns | 50 TOPS INT8 | DDR5 (shared) | ~2W | `amdxdna` |

All three share the same DDR5 memory controller. The hardware supports zero-copy — the kernel doesn't.

## Software Stack

```
┌──────────────────────────────────────────────────────┐
│                 User Application                      │
│  (HIP, OpenAI API, llama.cpp, flm)                   │
└────────┬────────────┬──────────────┬─────────────────┘
         │            │              │
    ┌────▼────┐ ┌─────▼──────┐ ┌────▼─────┐
    │  GPU    │ │   NPU     │ │   CPU    │
    │  ROCm   │ │ FastFlowLM│ │ llama.cpp│
    │  HIP    │ │ XRT SHIM  │ │          │
    └────┬────┘ └─────┬──────┘ └────┬─────┘
         │            │              │
    ┌────▼────┐ ┌─────▼──────┐ ┌────▼─────┐
    │amdgpu.ko│ │amdxdna.ko │ │  native  │
    │  TTM    │ │ IOMMU SVA │ │          │
    │  GTT    │ │  PASID    │ │          │
    └─────────┘ └───────────┘ └──────────┘
         │            │              │
    ┌────▼────────────▼──────────────▼─────┐
    │           IOMMU / SVA                │
    │       Shared Virtual Addressing      │
    └──────────────────────────────────────┘
```

## Key Integration Points

### 1. DMA-buf Zero-Copy (Proven ✅)

```
GPU allocates TTM_PL_TT (GTT, system RAM)
         │
    dma-buf export (drmPrimeHandleToFD)
         │
         ▼
NPU imports (DRM_IOCTL_PRIME_FD_TO_HANDLE)
         │
         ▼
Both access same physical pages: 27 GB/s read, 56 GB/s write
```

**Files:** `tests/test_gtt_dmabuf.cpp`, `tests/bench_gtt_dmabuf.cpp`

### 2. SMU Init Order Fix (PR'd — `lemonade-sdk/amdxdna-dkms#15`)

On Strix Halo, the NPU's SMU (System Management Unit) is embedded in the
firmware package loaded by PSP. The driver initialized SMU before PSP,
causing SMU init to always fail. The community workaround bypassed SMU
entirely, leaving the NPU without power management.

**Fix:** `patches/0001-amdxdna-fix-smu-init-order-strix-halo.patch`
Swaps init order: PSP → SMU. Also adds `stop_psp_no_smu` cleanup label.

### 3. NPU IP Block (Compile-tested on kernel 7.0.0)

`patches/amdgpu_npu.c` — wraps amdxdna into amdgpu's IP block lifecycle:

| Callback | What it does |
|----------|-------------|
| `early_init` | Discover NPU PCI function (bus:00.1) |
| `sw_init` | Request firmware, map BARs, init ring |
| `hw_init` | PSP load → SMU power on → firmware alive check |
| `hw_fini` | Suspend firmware → stop mailbox → stop PSP |
| `sw_fini` | Free ring, release firmware, put PCI dev |

**Integration patch:** `patches/0002-add-npu-ip-block.patch` (6 files)

### 4. HIP NPU Shim (Builds and runs ✅)

`libhip_npu.so` — LD_PRELOAD library that makes HIP see the NPU:

```
Without shim:  hipGetDeviceCount → 1 (GPU only)
With shim:     hipGetDeviceCount → 2 (GPU 0 + NPU 1)
```

**File:** `rocm-npu/hip_npu.cpp`

### 5. Scheduler Daemon (Running on Strix Halo ✅)

`npu-gpu-cpud.py` — REST API gateway at port 8080:

| Endpoint | Description |
|----------|-------------|
| `GET /v1/health` | Device status and policy |
| `GET /v1/models` | Available models (proxied from NPU) |
| `POST /v1/chat/completions` | Routes to NPU/GPU/CPU |

**Policy:** `< 2B→NPU`, `2B-8B→GPU`, `> 8B→CPU`

## Benchmarks

| Metric | Value |
|--------|-------|
| GPU↔NPU bandwidth (read) | 27 GB/s |
| GPU↔NPU bandwidth (write) | 56 GB/s |
| NPU prefill (qwen3:0.6b) | 50-93 tok/s |
| NPU decode (qwen3:0.6b) | 93-94 tok/s |
| NPU TTFT | 0.52-0.66s |
| NPU power | ~2W |

## Open PRs

| # | Repo | What | Link |
|---|------|------|------|
| 15 | `lemonade-sdk/amdxdna-dkms` | SMU init order fix | [PR #15](https://github.com/lemonade-sdk/amdxdna-dkms/pull/15) |
| 2474 | `lemonade-sdk/lemonade` | vLLM dual-GPU fix | [PR #2474](https://github.com/lemonade-sdk/lemonade/pull/2474) |
| 2459 | `lemonade-sdk/lemonade` | get_rocm_arch() fix | [PR #2459](https://github.com/lemonade-sdk/lemonade/pull/2459) |

## Repos

| Repo | URL | Description |
|------|-----|-------------|
| npu-gpu-cpu | https://github.com/bong-water-water-bong/npu-gpu-cpu | Main project — all patches, tests, daemon |
| 1bit-lemonade | https://github.com/bong-water-water-bong/1bit-lemonade | Fork with vLLM fix merged |
| amdxdna-dkms (fork) | https://github.com/bong-water-water-bong/amdxdna-dkms | Fork with SMU fix |
