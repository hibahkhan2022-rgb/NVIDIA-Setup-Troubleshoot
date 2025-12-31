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
