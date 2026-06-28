#!/usr/bin/env python3
"""
npu-gpu-cpud — Unified Control Plane Daemon

Routes inference requests to NPU, GPU, or CPU based on policy.

Policy (model_size → device):
  < 2B params  → NPU (lowest power)
  2B-8B params → GPU (fastest compute)
  > 8B params  → CPU (llama.cpp, fallback)

API:
  POST /v1/chat/completions  — OpenAI-compatible
  GET  /v1/health            — Device status

Usage:
  sudo ./npu-gpu-cpud.py [--port PORT]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

class Device:
    def __init__(self, name: str, device_type: str, available: bool):
        self.name = name
        self.type = device_type  # 'npu', 'gpu', 'cpu'
        self.available = available

def detect_devices() -> list[Device]:
    devices = []

    # CPU always available
    devices.append(Device("AMD Zen 5", "cpu", True))

    # NPU: check /dev/accel/accel0
    npu_avail = os.path.exists("/dev/accel/accel0")
    devices.append(Device("AMD XDNA 2 NPU (aie2)", "npu", npu_avail))

    # GPU: check via rocminfo
    gpu_avail = False
    try:
        result = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=5
        )
        gpu_avail = "gfx1151" in result.stdout or "Agent" in result.stdout
    except Exception:
        pass
    devices.append(Device("AMD Radeon 8060S (gfx1151)", "gpu", gpu_avail))

    return devices

# ---------------------------------------------------------------------------
# Policy engine
# ---------------------------------------------------------------------------

def select_device(model_size_b: float, devices: list[Device]) -> Optional[Device]:
    """Select the best device for a model based on parameter count."""
    npu = next((d for d in devices if d.type == "npu"), None)
    gpu = next((d for d in devices if d.type == "gpu"), None)
    cpu = next((d for d in devices if d.type == "cpu"), None)

    if model_size_b < 2 and npu and npu.available:
        return npu
    if model_size_b <= 8 and gpu and gpu.available:
        return gpu
    if cpu and cpu.available:
        return cpu
    return None

# ---------------------------------------------------------------------------
# Inference backends
# ---------------------------------------------------------------------------

def run_npu(model_name: str, prompt: str) -> str:
    """Run inference on NPU via FastFlowLM."""
    try:
        result = subprocess.run(
            ["flm", "run", model_name, "--prompt", prompt],
            capture_output=True, text=True, timeout=120
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        return "Error: NPU inference timed out"
    except Exception as e:
        return f"Error: NPU inference failed: {e}"

def run_gpu(model_name: str, prompt: str) -> str:
    """Run inference on GPU via Lemonade / llama.cpp ROCm."""
    try:
        # Use lemonade CLI if available, otherwise use the REST API
        result = subprocess.run(
            ["lemonade", "run", model_name, "--prompt", prompt],
            capture_output=True, text=True, timeout=120
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        return "Error: GPU inference timed out"
    except Exception as e:
        return f"Error: GPU inference failed: {e}"

def run_cpu(model_name: str, prompt: str) -> str:
    """Run inference on CPU via llama.cpp CPU mode."""
    try:
        result = subprocess.run(
            ["lemonade", "run", model_name, "--backend", "cpu", "--prompt", prompt],
            capture_output=True, text=True, timeout=300
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        return "Error: CPU inference timed out"
    except Exception as e:
        return f"Error: CPU inference failed: {e}"

# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

devices = detect_devices()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/v1/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            status = {
                "status": "ok",
                "devices": [
                    {"name": d.name, "type": d.type, "available": d.available}
                    for d in devices
                ],
                "policy": {
                    "< 2B params": "npu",
                    "2B-8B params": "gpu",
                    "> 8B params": "cpu",
                },
            }
            self.wfile.write(json.dumps(status, indent=2).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                req = json.loads(body)
            except json.JSONDecodeError:
                self.send_error(400, "Invalid JSON")
                return

            model = req.get("model", "unknown")
            messages = req.get("messages", [])
            prompt = messages[-1]["content"] if messages else ""

            # Estimate model size from name
            model_size_b = estimate_model_size(model)

            # Select device
            device = select_device(model_size_b, devices)
            if not device:
                self.send_error(503, "No available device")
                return

            # Run inference
            if device.type == "npu":
                output = run_npu(model, prompt)
            elif device.type == "gpu":
                output = run_gpu(model, prompt)
            else:
                output = run_cpu(model, prompt)

            # Return OpenAI-compatible response
            response = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": output,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "x-device": device.type,
                "x-device-name": device.name,
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response, indent=2).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {args[0]} {args[1]} {args[2]}")

def estimate_model_size(model_name: str) -> float:
    """Rough estimate of model size in billions of parameters."""
    model_lower = model_name.lower()
    for pattern, size in [
        ("0.5b", 0.5), ("1b", 1), ("1.5b", 1.5),
        ("2b", 2), ("3b", 3), ("4b", 4), ("7b", 7),
        ("8b", 8), ("9b", 9), ("13b", 13), ("14b", 14),
        ("20b", 20), ("34b", 34), ("70b", 70),
    ]:
        if pattern in model_lower:
            return size
    return 7  # default: assume 7B

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NPU+GPU+CPU Control Plane Daemon")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    args = parser.parse_args()

    print("=" * 60)
    print("  NPU + GPU + CPU = Unified Control Plane")
    print("=" * 60)
    print()
    print("Detected devices:")
    for d in devices:
        status = "✅" if d.available else "❌"
        print(f"  {status} {d.name:35s} ({d.type})")
    print()

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Listening on http://0.0.0.0:{args.port}")
    print(f"  GET  /v1/health")
    print(f"  POST /v1/chat/completions")
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.server_close()

if __name__ == "__main__":
    main()
