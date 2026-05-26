#!/bin/bash

# =================================================================
# Script setup Server cho Root (Python 3.12 + PyTorch CUDA)
# =================================================================

echo ">>> Bắt đầu cập nhật hệ thống (Quyền Root)..."
apt update && apt upgrade -y

echo ">>> Cài đặt các công cụ hỗ trợ..."
apt install unzip tmux wget git build-essential -y
apt-get install -y libgl1 2>&1 | tail -3
apt-get install -y libglib2.0-0 libgl1-mesa-dev libglx-mesa0 libgl1

echo ">>> Cài đặt Miniconda..."
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

# Khởi tạo conda cho shell hiện tại
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init bash

echo ">>> Chấp nhận điều khoản Anaconda..."
conda tos accept --channel https://repo.anaconda.com/pkgs/main
conda tos accept --channel https://repo.anaconda.com/pkgs/r

echo ">>> Tạo môi trường Conda py3.12..."
conda create -n py3.12 python=3.12 -y

echo ">>> Kích hoạt môi trường và cài đặt PyTorch..."
# Cần dùng lệnh này để activate ngay trong script
source ~/miniconda3/bin/activate py3.12

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "================================================================="
echo "SETUP HOÀN TẤT VỚI QUYỀN ROOT!"
echo "Hãy gõ: source ~/.bashrc để cập nhật môi trường."
echo "================================================================="