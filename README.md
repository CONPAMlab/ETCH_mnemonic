# Efficient Temporal Coding – YOLOv10 Based

## Overview
This project replicates the full functionality of the YOLOv10 model on example video clips to characterize object-level dynamics in naturalistic input.  
This serves as a first step toward testing the Efficient Temporal Coding Hypothesis and establishing a computational foundation for mnemonic representation.

---

## Environment Setup 

```bash
# Create and activate a Conda environment
conda create -n yolov10 python=3.10 -y
conda activate yolov10

# Clone the official YOLOv10 repository
git clone https://github.com/THU-MIG/yolov10.git
cd yolov10

# Install dependencies
pip install -r requirements.txt

# Install OpenCV with ffmpeg support
conda install -c conda-forge opencv

# Go back to your main project directory
cd ..
