# Building and Testing

## Prerequisites

- AMD Strix Halo system (Ryzen AI MAX+ 395 or similar)
- Ubuntu 26.04+ with kernel 7.0.0+
- ROCm 7.1+ (`sudo apt install rocm-dev hipcc`)
- FastFlowLM (`pip install fastflowlm` or `apt install flm`)
- libdrm-dev (`sudo apt install libdrm-dev libdrm-amdgpu-dev`)

## Quick Start

```bash
# Clone the repo
git clone https://github.com/bong-water-water-bong/npu-gpu-cpu.git
cd npu-gpu-cpu

# 1. Verify NPU is detected
sudo flm validate
# Expected: /dev/accel/accel0 with 8 columns, FW 1.1.2.65

# 2. Run the dma-buf zero-copy test
cd tests
g++ -std=gnu++17 -o test_gtt_dmabuf test_gtt_dmabuf.cpp \
    -I/usr/include/libdrm -ldrm -ldrm_amdgpu
sudo ./test_gtt_dmabuf
# Expected: PASS — GPU and NPU access same physical pages

# 3. Run the bandwidth benchmark
g++ -std=gnu++17 -O2 -o bench_gtt_dmabuf bench_gtt_dmabuf.cpp \
    -I/usr/include/libdrm -ldrm -ldrm_amdgpu
sudo ./bench_gtt_dmabuf
# Expected: 27 GB/s read, 56 GB/s write

# 4. Run the NPU device test
g++ -std=gnu++17 -o test_npu_dev test_npu_dev.cpp
./test_npu_dev
# Expected: NPU buffer alloc + dma-buf export OK

# 5. Build the HIP NPU shim
cd ../rocm-npu
cmake -S . -B build -DHIP_INCLUDE_DIR=/usr/include
cmake --build build
# Produces: build/libhip_npu.so

# 6. Test HIP device enumeration
cd ../tests
hipcc -o hip_list_devices hip_list_devices.cpp
LD_PRELOAD=../rocm-npu/build/libhip_npu.so ./hip_list_devices
# Expected: 2 devices — GPU 0 + NPU 1

# 7. Start the scheduler daemon
cd ../daemon
python3 npu-gpu-cpud.py --port 8080 --no-auto
# In another terminal:
curl http://localhost:8080/v1/health
curl http://localhost:8080/v1/chat/completions \
  -d '{"model":"qwen3:0.6b","messages":[{"role":"user","content":"Hi"}]}'

# 8. Run unified benchmark
cd ../tests
python3 bench_unified.py
```

## Kernel Module Build

```bash
# Requires full kernel source tree
cd /usr/src/linux-source-7.0.0  # or your kernel tree

# Apply NPU IP block patches
cp patches/amdgpu_npu.c     drivers/gpu/drm/amd/amdgpu/
cp patches/amdgpu_npu.h     drivers/gpu/drm/amd/amdgpu/
cp patches/amdgpu_npu_mgr.c drivers/gpu/drm/amd/amdgpu/
cp patches/amdgpu_npu_sched.c drivers/gpu/drm/amd/amdgpu/

# Apply integration patch
git am patches/0002-add-npu-ip-block.patch

# Build just the amdgpu module
make -C /lib/modules/$(uname -r)/build \
  M=$PWD/drivers/gpu/drm/amd/amdgpu modules -j$(nproc)
```

## Testing on Different Hardware

| Test | Strix Halo | Phoenix | Phoenix2 |
|------|-----------|---------|----------|
| dma-buf zero-copy | ✅ Tested | Untested | Untested |
| NPU inference | ✅ 93 tok/s | Untested | Untested |
| SMU init fix | ✅ Tested | Should work | Should work |
| HIP shim | ✅ 2 devices | Untested | Untested |

## CI

Tests require a self-hosted Strix Halo runner with:
- `/dev/accel/accel0` — NPU device
- ROCm 7.1+ — GPU compute
- FastFlowLM — NPU inference

See `.github/workflows/` for CI configuration (TBD).
