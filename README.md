# Face Alignment with HRNet

This project implements **face landmark detection and face alignment** using the HRNet architecture on the 300W dataset.

---

## 📌 Overview

* Model: HRNet (High-Resolution Network)
* Task: 68-point facial landmark detection
* Dataset: 300W (DeepLake version)
* Output:

  * Landmark heatmaps
  * 68 keypoint predictions
  * Aligned face images

---

## ⚙️ Environment Setup

```bash
conda create -n face_landmark python=3.10 -y
conda activate face_landmark

pip install "deeplake<4"
pip install opencv-python-headless

pip install torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

---

## 📂 Dataset

We use the DeepLake version of the 300W dataset:

```
hub://activeloop/300w
```

Each sample contains:

* `images`: raw image
* `keypoints`: 68 landmarks (x, y, visibility)
* `labels`: train/test split

### Data Preprocessing

* Crop face using landmarks (scale = 1.5)
* Resize image to `256 × 256`
* Convert keypoints to heatmaps (`64 × 64`)
* Normalize pixel values to `[0, 1]`

---

## 🧠 Model

HRNet maintains high-resolution feature representations throughout the network and performs multi-scale fusion.

Compared to traditional CNNs:

* No heavy downsampling
* Better spatial precision
* More accurate keypoint localization

---

## 🚀 Training

### Main Settings

| Parameter    | Value    |
| ------------ | -------- |
| Input size   | 256×256  |
| Heatmap size | 64×64    |
| Landmarks    | 68       |
| Batch size   | 8        |
| Optimizer    | Adam     |
| Loss         | MSE Loss |
| Device       | GPU      |

### Training Strategy

* Stage 1:

  * Epochs: 0 → 200
  * Learning rate: 1e-3

* Stage 2 (fine-tuning):

  * Epochs: 200 → 300
  * Learning rate: 1e-4

---

## 📊 Evaluation Metrics

* Mean NME (Normalized Mean Error)
* Median NME
* Failure Rate @ 0.08
* AUC @ 0.08

Also includes **per-region error analysis**:

* Jaw
* Eyebrows
* Eyes
* Nose
* Mouth

---

## 📈 Results

| Metric       | Value  |
| ------------ | ------ |
| Mean NME     | 0.0939 |
| Median NME   | 0.0794 |
| Failure@0.08 | 47.67% |
| AUC@0.08     | 0.1028 |

### Per-region Error

| Region        | Error |
| ------------- | ----- |
| Jaw           | 9.81  |
| Right Eyebrow | 5.89  |
| Left Eyebrow  | 6.25  |
| Nose          | 4.21  |
| Right Eye     | 4.39  |
| Left Eye      | 4.42  |
| Mouth         | 5.86  |

---

## 🖼 Visualization

* Green points: Ground truth
* Red points: Prediction

Example results can be found in:

```
test/full_eval_epoch_200/
```

---

## 🔧 Inference & Face Alignment

Pipeline:

```
Image → HRNet → Heatmap → 68 landmarks → 5 keypoints → Affine Transform → Aligned face
```

Generated outputs include:

* Original image
* Cropped face
* Predicted landmarks
* Aligned face

---

## 📁 Project Structure

```
FaceAlignment/
│
├── models/
├── datasets/
├── utils/
├── train.py
├── eval.py
├── predict_single_from_dataset.py
├── config.py
├── README.md
```

---

## 📌 Notes

* Dataset and checkpoints are not included
* GPU is recommended for training
* Small dataset may lead to overfitting

---

## 📎 References

* HRNet: High-Resolution Representations for Visual Recognition
* 300W Dataset
