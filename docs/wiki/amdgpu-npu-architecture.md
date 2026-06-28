# AMD NPU in amdgpu — Architecture

## Why fold NPU into amdgpu?

The NPU on Strix Halo shares the same memory controller and PCIe root complex
as the GPU. Keeping a separate driver creates duplicate memory management,
separate device nodes, and no zero-copy path.

## IP Integration Pattern

amdgpu already handles heterogeneous IP blocks:

| IP Block | File | Purpose |
|----------|------|---------|
| GFX | `amdgpu_gfx.c` | 3D/compute shaders |
| SDMA | `amdgpu_sdma.c` | DMA engines |
| VCN | `amdgpu_vcn.c` | Video encode/decode |
| JPEG | `amdgpu_jpeg.c` | JPEG encode/decode |
| **NPU** | **`amdgpu_npu.c`** | **AI inference (new)** |

Lifecycle: `early_init → sw_init → hw_init → hw_fini → sw_fini`

## Memory Model

```
TTM_PL_VRAM  → GPU accessible
TTM_PL_NPU   → NPU accessible  (new)
TTM_PL_TT    → CPU accessible (GTT)
```

Because all three share physical RAM, migration between domains is a
page-table update, not a data copy.

## Firmware

Loaded via `request_firmware("amdgpu/npu_<asic>_<rev>.bin")`.
ABI reverse-engineered from `drivers/staging/amdxdna/`.
