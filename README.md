# Efficient Temporal Coding – Object Dynamics Extraction Pipeline
This project implements a complete object-level visual dynamics extraction system using **Ultralytics YOLOv11** and **SORT multi-object tracking**.  

It is designed for large-scale video datasets such as **Ego4D**, and supports:
- Object detection  
- Class labeling  
- Persistent Track IDs  
- Class-colored bounding boxes  
- Object cropping  
- Frame-level summary statistics  
The pipeline supports cognitive science applications including the **Efficient Temporal Coding Hypothesis**, visual working memory research, naturalistic visual input analysis, and large-scale perceptual statistics extraction.
---

## 1. Environment Setup

### Create a new Conda environment
conda create -n yolov10 python=3.10 -y
conda activate yolov10

### Install required packages
pip install ultralytics
pip install filterpy
pip install numpy<2
pip install opencv-python
pip install pandas
pip install tqdm


### (Optional but recommended on Mac M1/M2/M3)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
### Or for accelerated inference (Mac only):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu-mps

---

## 2. Project Structure

Your project folder should look like this:

```text
NVSExtracter/
│
├── main.py
├── sort.py                  # not used!
├── tracker.py               # not used!
├── videos/                # (Place your input videos here)
├── weights/
│   └── yolo11n.pt        # auto-downloaded if missing
│
├── outputs/
│   ├── xxx_annotated.mp4
│   ├── crops/
│   └── detections_summary.csv
│
└── README.md
```
---

## 3. Running the Pipeline

Place any .mp4 / .mov / .avi videos into:

videos/

Then run:
```
python main.py
```
After processing, results appear in:

```text
outputs/
│
├── <video>_annotated.mp4       # Video with class labels + track IDs + colored boxes
├── crops/                      # Folder containing object crops per frame
│     ├── <video>_track3_frame27.jpg
│     ├── <video>_track3_frame28.jpg
│     └── ...
└── detections_summary.csv      # Frame-by-frame summary statistics
```

---

## 4. What the Pipeline Produces

### 1). Annotated Videos
- Bounding boxes with **consistent class colors**
- Class labels (e.g., person, car, dog)
- Persistent **SORT Track IDs**
- Real-time object tracking visualization

Example label on video:
```
person | ID 7
```

### 2). Object Crops
For each detected object, the pipeline saves a cropped image:
- Cropped image saved per frame
- Automatically resized if too large
- Useful for downstream appearance / motion analysis

### 3). CSV Summary
detections_summary.csv includes:
- video (filename)
- frame index
- n_objects (count)
- mean_area (bounding box area avg)
- mean_confidence (YOLO confidence avg)
---

## 5. Research Integration

This system is optimized for:
- Naturalistic vision analysis (Ego4D, video corpora)
- Visual working memory and object persistence studies
- Efficient Temporal Coding Hypotheses
- Linking low-level statistics → VWM precision & stability
- Preprocessing for computational models (saliency, compression, TCC)
- Large-scale behavioral + vision joint modeling

---

## 6. Notes on Detection Quality

YOLO11n (nano) is extremely fast but less accurate.
If your videos have:
- Small objects
- Fast motion
- Low lighting
- Clutter
- High resolution scenes

Consider upgrading:

* Nano (fastest, weakest)
* Small
* Medium
* Large (best accuracy)

To upgrade weights, edit in main.py:

CONFIG["model"]["weights"] = "weights/yolo11s.pt"

Download from:
https://github.com/ultralytics/ultralytics/releases

---

## 7. Troubleshooting

### No detections in video
- Try larger YOLO model (yolo11s, yolo11m)
- Increase resolution (change LetterBox size)
- Lower confidence threshold (e.g., conf=0.15)

### Slow performance
- Use device="mps" on Mac for hardware acceleration
- Use "device": "cuda" if a GPU exists
- Use YOLO nano models

### SORT tracker jumps
- Increase iou_threshold in CONFIG
- Increase min_hits

---

## 8. Citation
If you use this pipeline in academic work:

Ultralytics YOLOv11
https://docs.ultralytics.com/

SORT Tracker (Bewley et al., 2016)
https://arxiv.org/abs/1602.00763

---

## 9. Contact
For Ego4D-scale batch runners, optical flow modules, scene complexity features, or memory-model alignment, feel free to ask!

contact: sliu485@ucr.edu

