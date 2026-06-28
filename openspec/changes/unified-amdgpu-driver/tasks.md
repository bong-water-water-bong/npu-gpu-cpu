# Tasks — Unified AMDGPU Driver

## Phase 0: Shared GTT (dma-buf) Prototype

- [x] **0.1** Write dma-buf sharing test: amdgpu TTM_PL_TT → export → amdxdna import
  Test: [`tests/test_gtt_dmabuf.cpp`](../../tests/test_gtt_dmabuf.cpp)
- [x] **0.2** Verify zero-copy: GPU writes pattern, NPU reads it back
  Result: ✅ PASS — GPU wrote 0xDEADBEEF, NPU read same; NPU wrote 0xCAFEBABE, GPU read same
- [ ] **0.3** Measure: bandwidth comparison (GTT→NPU vs VRAM→CPU→NPU)

## Phase 1: Kernel

- [x] **1.0** Fix SMU init order in aie2_hw_start() — PSP before SMU (Strix Halo)
  Patch: [`patches/0001-amdxdna-fix-smu-init-order-strix-halo.patch`](../../patches/0001-amdxdna-fix-smu-init-order-strix-halo.patch)
  DKMS PR: [lemonade-sdk/amdxdna-dkms#15](https://github.com/lemonade-sdk/amdxdna-dkms/pull/15)
- [ ] **1.1** Read amdxdna staging driver — MMIO layout, firmware ABI, ring protocol
- [ ] **1.2** Read amdgpu IP block init — how VCN/SDMA/JPEG are initialized
- [ ] **1.3** Write `amdgpu_npu_early_init()` — PCI BAR discovery
- [ ] **1.4** Write `amdgpu_npu_init()` — firmware load, ring init
- [ ] **1.5** Add `TTM_PL_NPU` to TTM memory manager
- [ ] **1.6** Register `AMDGPU_NPU_CTX` IOCTL
- [ ] **1.7** Write `amdgpu_npu_sched.c` — NPU ring submission
- [ ] **1.8** Add NPU PCI DID to driver table
- [ ] **1.9** Build-test kernel module

## Phase 2: ROCm Userspace

- [ ] **2.1** Extend `rocminfo` to detect NPU agent
- [ ] **2.2** Write `hip_npu.cpp` — device enumeration
- [ ] **2.3** Write `hip_npu_memory.cpp` — hipMalloc/hipFree
- [ ] **2.4** Write NPU AQL packet defs (`npu_aql.h`)
- [ ] **2.5** Implement `hipLaunchKernel` for NPU
- [ ] **2.6** Test: simple GEMM on NPU via HIP

## Phase 3: Scheduler Daemon

- [ ] **3.1** Write `npu-gpu-cpud` — gRPC/REST API
- [ ] **3.2** Implement dispatch policy table
- [ ] **3.3** Integrate with Lemonade BackendManager

## Documentation

- [ ] **4.1** Write `docs/wiki/amdgpu-npu-architecture.md`
- [ ] **4.2** Write `docs/wiki/building-and-testing.md`
