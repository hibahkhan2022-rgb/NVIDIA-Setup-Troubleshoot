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
How to access Docker Container:
`
sudo docker run --rm -it --runtime nvidia --network host \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/jetson/jetson-playground/jetson-bench:/workspace \
  -w /workspace \
  nvcr.io/nvidia/pytorch:24.11-py3-igpu


`
