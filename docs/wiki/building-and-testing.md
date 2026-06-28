# Building and Testing

## Prerequisites

- Linux kernel source tree with `amdgpu` enabled
- ROCm toolchain (`hipcc`, `rocminfo`)
- Strix Halo hardware (gfx1151)

## Build Kernel Module

```bash
make -C /lib/modules/$(uname -r)/build M=$(pwd)/drivers/gpu/drm/amd/amdgpu modules
sudo rmmod amdxdna && sudo rmmod amdgpu
sudo modprobe amdgpu
cat /sys/kernel/debug/dri/0/amdgpu_npu_info
```

## Test Memory Sharing

```bash
./tests/test_shared_bo --size 1G --domain NPU --peer GPU
```

## Test ROCm HIP

```bash
rocminfo | grep -A5 NPU
hipcc -o tests/gemm_npu tests/gemm_npu.cpp
./tests/gemm_npu --device npu --m 1024 --n 1024 --k 1024
```
