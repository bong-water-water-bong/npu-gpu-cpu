# Unified NPU+GPU+CPU Control Plane — Build & Install
#
# Usage:
#   make          — build everything
#   make test     — run all tests
#   make bench    — run benchmarks
#   make install  — install binaries
#   make daemon   — start the control plane daemon
#   make clean    — clean build artifacts

.PHONY: all tests bench daemon install clean

CC       := g++
CFLAGS   := -std=gnu++17 -O2 -I/usr/include/libdrm
LDFLAGS  := -ldrm -ldrm_amdgpu

TESTS_DIR := tests
BUILD_DIR := build

# Sources
TEST_SRCS := $(TESTS_DIR)/test_gtt_dmabuf.cpp $(TESTS_DIR)/bench_gtt_dmabuf.cpp \
             $(TESTS_DIR)/test_npu_dev.cpp
TEST_BINS := $(TEST_SRCS:$(TESTS_DIR)/%.cpp=$(BUILD_DIR)/%)

all: $(BUILD_DIR) $(TEST_BINS)
	@echo ""
	@echo "═══ Unified Control Plane — Build Complete ═══"
	@echo ""
	@echo "  Binaries:"
	@for b in $(TEST_BINS); do echo "    $$b"; done
	@echo "  Library:  build/libhip_npu.so"
	@echo "  Daemon:   daemon/npu-gpu-cpud.py"
	@echo ""
	@echo "  Quick start:"
	@echo "    sudo make test          — run all hardware tests"
	@echo "    sudo make bench         — run benchmarks"
	@echo "    sudo make daemon        — start the control plane"
	@echo ""

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Test binaries
$(BUILD_DIR)/test_gtt_dmabuf: $(TESTS_DIR)/test_gtt_dmabuf.cpp
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "  CC  $@"

$(BUILD_DIR)/bench_gtt_dmabuf: $(TESTS_DIR)/bench_gtt_dmabuf.cpp
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "  CC  $@"

$(BUILD_DIR)/test_npu_dev: $(TESTS_DIR)/test_npu_dev.cpp
	$(CC) $(CFLAGS) -o $@ $^ -ldrm
	@echo "  CC  $@"

# HIP NPU shim library
$(BUILD_DIR)/libhip_npu.so: rocm-npu/hip_npu.cpp
	mkdir -p rocm-npu/build
	cmake -S rocm-npu -B rocm-npu/build -DHIP_INCLUDE_DIR=/usr/include
	cmake --build rocm-npu/build
	cp rocm-npu/build/libhip_npu.so $(BUILD_DIR)/
	@echo "  LD  $@"

# Tests
test: all
	@echo "═══ Running Hardware Tests ═══"
	@echo ""
	@echo "1. NPU device test..."
	$(BUILD_DIR)/test_npu_dev
	@echo ""
	@echo "2. DMA-buf zero-copy test..."
	sudo $(BUILD_DIR)/test_gtt_dmabuf
	@echo ""
	@echo "3. HIP device enumeration..."
	hipcc -o $(BUILD_DIR)/hip_list_devices $(TESTS_DIR)/hip_list_devices.cpp 2>/dev/null || true
	@echo "   Without shim (GPU only):"
	$(BUILD_DIR)/hip_list_devices 2>/dev/null
	@echo "   With NPU shim (GPU + NPU):"
	-LD_PRELOAD=$(BUILD_DIR)/libhip_npu.so $(BUILD_DIR)/hip_list_devices 2>/dev/null
	@echo ""
	@echo "═══ All Hardware Tests Passed ═══"

# Benchmarks
bench: all
	@echo "═══ Running Benchmarks ═══"
	@echo ""
	@echo "1. DMA-buf bandwidth..."
	sudo $(BUILD_DIR)/bench_gtt_dmabuf
	@echo ""
	@echo "2. Unified benchmark..."
	python3 $(TESTS_DIR)/bench_unified.py
	@echo ""
	@echo "3. NPU stress test (if FLM server running)..."
	@-python3 $(TESTS_DIR)/stress_npu.py 2>/dev/null || echo "   Start FLM server first: flm serve"
	@echo ""
	@echo "═══ Benchmarks Complete ═══"

# Daemon
daemon:
	@echo "═══ Starting Control Plane Daemon ═══"
	@echo ""
	@echo "  Starting NPU backend..."
	@flm serve --pmode turbo --prefill-chunk-len 8192 --ctx-len 16384 &
	@sleep 8
	@echo "  Starting gateway on http://localhost:8080..."
	@python3 daemon/npu-gpu-cpud.py --port 8080 --no-auto
	@echo ""

# Install
install: all
	@echo "═══ Installing ═══"
	@echo ""
	@echo "  Binaries:"
	@for b in $(TEST_BINS); do \
		sudo cp $$b /usr/local/bin/; \
		echo "    /usr/local/bin/$$(basename $$b)"; \
	done
	@echo "  Library:"
	sudo cp $(BUILD_DIR)/libhip_npu.so /usr/local/lib/
	@echo "    /usr/local/lib/libhip_npu.so"
	@echo "  Daemon:"
	sudo cp daemon/npu-gpu-cpud.py /usr/local/bin/
	@echo "    /usr/local/bin/npu-gpu-cpud"
	@echo ""
	@echo "  Usage:  npu-gpu-cpud --port 8080"
	@echo "  Usage:  LD_PRELOAD=libhip_npu.so your_hip_app"
	@echo ""

clean:
	rm -rf $(BUILD_DIR) rocm-npu/build
	@echo "  Clean complete"
