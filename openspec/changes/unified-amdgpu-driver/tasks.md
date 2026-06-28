# Tasks — Unified AMDGPU Driver

## Phase 0: Shared GTT (dma-buf) Prototype

- [x] **0.1** Write dma-buf sharing test: amdgpu TTM_PL_TT → export → amdxdna import
  Test: [`tests/test_gtt_dmabuf.cpp`](../../tests/test_gtt_dmabuf.cpp)
- [x] **0.2** Verify zero-copy: GPU writes pattern, NPU reads it back
  Result: ✅ PASS — GPU wrote 0xDEADBEEF, NPU read same; NPU wrote 0xCAFEBABE, GPU read same
- [x] **0.3** Measure: bandwidth comparison (GTT→NPU vs GPU local)
  Test: [`tests/bench_gtt_dmabuf.cpp`](../../tests/bench_gtt_dmabuf.cpp)
  Result: NPU read 27 GB/s, NPU write 56 GB/s — **identical to GPU local access**
  Notes: VRAM→CPU→NPU copy path not applicable on APU (no discrete VRAM)

## Phase 1: Kernel

- [x] **1.0** Fix SMU init order in aie2_hw_start() — PSP before SMU (Strix Halo)
  Patch: [`patches/0001-amdxdna-fix-smu-init-order-strix-halo.patch`](../../patches/0001-amdxdna-fix-smu-init-order-strix-halo.patch)
  DKMS PR: [lemonade-sdk/amdxdna-dkms#15](https://github.com/lemonade-sdk/amdxdna-dkms/pull/15)
- [ ] **1.1** Read amdxdna staging driver — MMIO layout, firmware ABI, ring protocol
- [x] **1.2** Read amdgpu IP block init — how VCN/SDMA/JPEG are initialized
  Result: `amdgpu_ip_block_add()` + `amdgpu_device_ip_init()` pattern documented
- [x] **1.3** Write `amdgpu_npu_early_init()` — PCI BAR discovery
  File: [`patches/amdgpu_npu.c`](../../patches/amdgpu_npu.c)
- [x] **1.4** Write `amdgpu_npu_init()` — firmware load, ring init
  File: [`patches/amdgpu_npu.c`](../../patches/amdgpu_npu.c) — includes sw_init, hw_init
- [x] **1.5** Add NPU GTT sub-allocator (`amdgpu_npu_mgr.c`)
  Note: TTM_PL_NPU not added — all 9 TTM slots are full. Use GTT + dma-buf instead.
  File: [`patches/amdgpu_npu_mgr.c`](../../patches/amdgpu_npu_mgr.c)
- [x] **1.6** Register `AMDGPU_NPU_CTX` IOCTL
  File: [`patches/amdgpu_npu.c`](../../patches/amdgpu_npu.c) — includes ctx + exec ioctls
- [x] **1.7** Write `amdgpu_npu_sched.c` — NPU ring submission
  File: [`patches/amdgpu_npu_sched.c`](../../patches/amdgpu_npu_sched.c)
- [x] **1.8** Add NPU PCI DID to driver table
  Patch: [`patches/0002-add-npu-ip-block.patch`](../../patches/0002-add-npu-ip-block.patch)
- [x] **1.9** Build script
  Script: [`patches/build.sh`](../../patches/build.sh)
  (Full build-test requires kernel source tree with amdgpu; not available in this env)

## Phase 2: ROCm Userspace

- [x] **2.1** Extend `rocminfo` to detect NPU agent
  Note: Already done by ROCm — `rocminfo` shows Agent 3: `aie2`, DSP type
- [x] **2.2** Write `hip_npu.cpp` — device enumeration
  File: [`rocm-npu/hip_npu.cpp`](../../rocm-npu/hip_npu.cpp)
  Result: `LD_PRELOAD=libhip_npu.so` — HIP sees 2 devices (GPU + NPU)
- [x] **2.3** Write `hip_npu_memory.cpp` — hipMalloc/hipFree
  (Merged into hip_npu.cpp — mmap-based GTT allocation)
- [x] **2.4** Write NPU AQL packet defs (`npu_aql.h`)
  File: [`rocm-npu/npu_aql.h`](../../rocm-npu/npu_aql.h)
- [ ] **2.5** Implement `hipLaunchKernel` for NPU
  (Needs NPU firmware integration — mailbox-based command submission)
- [ ] **2.6** Test: simple GEMM on NPU via HIP
  (Blocked on 2.5 — needs working kernel dispatch)

## Phase 3: Scheduler Daemon

- [x] **3.1** Write `npu-gpu-cpud` — REST API + health endpoint
  File: [`daemon/npu-gpu-cpud.py`](../../daemon/npu-gpu-cpud.py)
- [x] **3.2** Implement dispatch policy table
  Policy: `< 2B → NPU`, `2B-8B → GPU`, `> 8B → CPU`
- [ ] **3.3** Integrate with Lemonade BackendManager
  (Needs: actual NPU inference integration via flm/lemonade)

## Documentation

- [ ] **4.1** Write `docs/wiki/amdgpu-npu-architecture.md`
- [ ] **4.2** Write `docs/wiki/building-and-testing.md`
