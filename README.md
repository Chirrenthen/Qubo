# 🔐 Qubo  
**Smart Door Lock System powered by Raspberry Pi 5 + Arduino Uno**

A modular, edge-based access control system integrating:
- Face Recognition (InsightFace + ONNX)
- Web Dashboard (real-time control & logs)
- Serial Communication (Arduino lock control)
- Camera-based authentication (Picamera2)

---

## 🚀 Architecture Overview

[Camera] → Raspberry Pi → Face Recognition → Decision Engine
↓
Serial (USB)
↓
Arduino Uno → Lock Mechanism
↑
Web Dashboard (HTTP)

## ⚙️ Setup Guide

### 📌 1. System Dependencies (Raspberry Pi)

Install all required low-level libraries (mandatory for OpenCV + camera stack):

```bash
sudo apt update && sudo apt upgrade -y

sudo apt install -y \
    python3-dev \
    build-essential \
    cmake \
    libatlas-base-dev \
    libjpeg-dev \
    libtiff5-dev \
    libopenjp2-7-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libcanberra-gtk* \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libcamera-dev \
    libcamera-apps \
    python3-picamera2
```
## 2. Project Setup (Virtual Environment)
```bash
# Create workspace
mkdir -p ~/qubo
cd ~/qubo

# Create virtual environment with system package access (IMPORTANT)
python3 -m venv venv --system-site-packages
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```
## Install Dependencies
```bash
pip install \
    numpy \
    pyserial \
    opencv-python \
    insightface==0.7.3 \
    onnxruntime
```

## Running
```bash
# Activate environment
source ~/qubo/venv/bin/activate

# Run the project (adjust port if needed)
python3 main.py /dev/ttyUSB0
```
## Web Interface

http://<raspberry-pi-ip>:7000
