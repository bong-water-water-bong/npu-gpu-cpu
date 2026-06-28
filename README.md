# NPU + GPU + CPU = Unified Control Plane

Single kernel driver + unified memory for AMD Strix Halo on Linux.

**Goal**: Fold `amdxdna` (XDNA 2 NPU) into `amdgpu` so the NPU, GPU, and CPU share one memory manager, one DRM file descriptor, and one ROCm compute API.

## Structure

```
├── AGENTS.md                          # Project standards
├── openspec/changes/
│   └── unified-amdgpu-driver/
│       ├── proposal.md                # Architecture proposal
│       └── tasks.md                   # Implementation tracking
├── docs/wiki/
│   ├── amdgpu-npu-architecture.md     # Durable architecture docs
│   └── building-and-testing.md        # Build guide
└── patches/                           # Kernel patches
```

## Status

**Phase 0: Planning** — Proposal written, awaiting Phase 1 (kernel module) work.

## Layers

| Layer | What | Where |
|-------|------|-------|
| Kernel | `amdgpu` NPU IP block | `drivers/gpu/drm/amd/amdgpu/` |
| Runtime | ROCm HIP NPU agent | `rocm-npu/` (this repo) |
| Scheduler | `npu-gpu-cpud` daemon | `daemon/` (this repo) |
