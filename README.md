# NVIDIA-Setup-Troubleshoot
This is a repo of NVIDIA setup instructions + troubleshoot. Refer for future errors. 

Materials required for basic setup:
1. Jetson Orin Nano
2. DSP-DSP Cable
3. Monitor (LG DP-friendly)
4. Ethernet cable
5. Power Supply (7 A, 45 W)
6. Keyboard, mouse initial setup
7. MicoSD card + reader

# Setup 

## Access the project directory
```
cd jetson-playground/jetson-bench
ls
```
You should see something callsed bench_torchvision.py

## How to access Docker Container:
```
sudo docker run --rm -it --runtime nvidia --network=host \
  -v "$PWD":/workspace \
  dustynv/l4t-pytorch:r36.4.0 \
  bash
```
Then, cd into the workspace. Should also see bench_torchvision.py. Every time you want to run the benchmark, write out:
```
python3 bench_torchvision.py
```
## Example Script
```
import time, statistics
import torch
import torchvision.models as models

torch.set_num_threads(1)

WARMUP = 10
ITERS = 200
RUNS = 10        # how many times to repeat each device
BATCH = 5
H, W = 480, 480

def bench(device: str):
    model = models.resnet18(weights=None).eval().to(device)
    x = torch.randn(BATCH, 3, H, W, device=device)

    # warmup
    for _ in range(WARMUP):
        with torch.no_grad():
            _ = model(x)
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    p50 = statistics.median(times)
    p95 = statistics.quantiles(times, n=20)[18]
    P99 = statistics.quantiles(times, n=100)[98]
    fps = 1000.0 / (sum(times) / len(times))
    return p50, p95, p99, fps

def main():
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print()

    for dev in ["cpu", "cuda"]:
        if dev == "cuda" and not torch.cuda.is_available():
            continue

        results = []
        print(f"=== {dev.upper()} ===")

        for r in range(RUNS):
            p50, p95, p99, fps = bench(dev)
            results.append((p50, p95, p99, fps))
            print(f"run {r+1}: p50={p50:.2f}ms | p95={p95:.2f}ms | p99={p99:.2f}ms | FPS={fps:.2f}")

        avg_p50 = sum(x[0] for x in results) / RUNS
        avg_p95 = sum(x[1] for x in results) / RUNS
        avg_p99 = sum(x[2] for x in results) / RUNS
        avg_fps = sum(x[3] for x in results) / RUNS

        print(f"AVG {dev.upper()}: p50={avg_p50:.2f}ms | p95={avg_p95:.2f}ms | p99={avg_p99:.2f}ms | FPS={avg_fps:.2f}")
        print()

if __name__ == "__main__":
    main()
```
