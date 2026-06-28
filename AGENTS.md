# AGENTS.md

## Project

Unified kernel driver + ROCm integration for Strix Halo NPU+GPU+CPU on Linux.

## Repo Structure

```
npu-gpu-cpu/
├── AGENTS.md              # This file — agent instruction schema
├── openspec/
│   └── changes/
│       └── <change-id>/
│           ├── proposal.md   # Architecture / design
│           └── tasks.md      # Implementation tracking
├── docs/
│   └── wiki/                 # Durable project knowledge
├── patches/
│   └── ...                   # Kernel patches against linux.git
├── rocm-npu/                 # ROCm runtime NPU target (userspace)
│   ├── hip_npu.cpp
│   └── CMakeLists.txt
└── tests/
    └── ...
```

## Workflow

1. Start non-trivial work with `openspec/changes/<change-id>/proposal.md`
2. Track implementation in `openspec/changes/<change-id>/tasks.md`
3. Update `docs/wiki/` whenever work reveals durable repo knowledge
4. Keep changes surgical, simple, and verified

## References

- `amdgpu` in-tree driver: `drivers/gpu/drm/amd/amdgpu/`
- `amdxdna` staging driver: `drivers/staging/amdxdna/`
- ROCm: `https://github.com/ROCm/`
