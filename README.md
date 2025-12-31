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
ITERS = 500
RUNS = 5        # how many times to repeat each device
BATCH = 5
H, W = 224,224

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
    p99 = statistics.quantiles(times, n=100)[98]
    fps = 1000.0 / (sum(times) / len(times)) * BATCH
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
### Results
```
=== CPU ===
run 1: p50=448.76ms | p95=743.73ms | p99=960.61ms | FPS=10.35
run 2: p50=466.60ms | p95=703.35ms | p99=955.82ms | FPS=9.96
run 3: p50=455.10ms | p95=574.30ms | p99=826.71ms | FPS=10.92
run 4: p50=410.71ms | p95=552.32ms | p99=746.41ms | FPS=11.67
run 5: p50=396.26ms | p95=490.85ms | p99=642.00ms | FPS=12.23
AVG CPU: p50=435.49ms | p95=612.91ms | p99=826.31ms | FPS=11.03

=== CUDA ===
run 1: p50=19.69ms | p95=20.65ms | p99=21.79ms | FPS=256.69
run 2: p50=19.88ms | p95=23.91ms | p99=26.84ms | FPS=245.56
run 3: p50=19.91ms | p95=23.97ms | p99=26.43ms | FPS=244.83
run 4: p50=19.69ms | p95=21.52ms | p99=24.32ms | FPS=253.60
run 5: p50=19.71ms | p95=21.70ms | p99=23.81ms | FPS=252.64
AVG CUDA: p50=19.78ms | p95=22.35ms | p99=24.64ms | FPS=250.66

```
