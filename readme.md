# 🧕 ML Hijab vs Non-Hijab Classification
> Auto folder-label · Multi-algorithm · Realtime Webcam + Video + Image

---

## 📁 File Structure

```
ML_hijab_nonhijab/
├── template_ml_image.ipynb   ← Training in Google Colab
├── app_streamlit.py          ← Streamlit implementation (web UI)
├── app_regular.py            ← Python OpenCV implementation (desktop)
└── README.md
```

---

## 🗂️ Dataset Structure (in Google Drive)

Simply create one folder per class — folder name = automatic label name.

```
dataset/
├── cat/
│   ├── image1.jpg
│   └── image2.jpg
├── dog/
│   ├── image1.jpg
│   └── image2.jpg
└── bird/          ← can add unlimited classes
    └── image1.jpg
```

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

---

## 🚀 Usage Steps

### Step 1 — Training in Google Colab

1. Upload `training_colab.ipynb` to Google Colab
2. Enable GPU: **Runtime → Change runtime type → T4 GPU**
3. Prepare dataset in Google Drive with the folder structure above
4. Edit path in **Cell 3 (Configuration)**:
   ```python
   DATASET_PATH = '/content/drive/MyDrive/dataset'  # adjust accordingly
   OUTPUT_PATH  = '/content/drive/MyDrive/ml_output'
   ```
5. Run all cells from top to bottom
6. Select algorithm when prompted:

| No | Algorithm     | Speed      | Accuracy | Best For              |
|----|---------------|------------|----------|-----------------------|
| 1  | SVM           | ⚡ Fast    | ⭐⭐⭐    | Small–medium dataset  |
| 2  | Random Forest | ⚡ Fast    | ⭐⭐⭐    | Diverse dataset       |
| 3  | KNN           | ⚡ Very fast | ⭐⭐   | Quick prototyping     |
| 4  | MobileNetV2   | 🔥 Medium  | ⭐⭐⭐⭐  | Mobile/edge deploy    |
| 5  | ResNet50      | 🔥 Slow    | ⭐⭐⭐⭐⭐ | High accuracy         |
| 6  | EfficientNetB0| 🔥 Medium  | ⭐⭐⭐⭐⭐ | Best balance          |

7. Download `ml_output/` folder from Google Drive to local

---

### Step 2 — Deploy Streamlit

```bash
# Install dependencies
pip install streamlit opencv-python scikit-learn tensorflow
pip install scikit-image streamlit-webrtc av pillow

# Ensure ml_output folder exists in the same directory
# Run
streamlit run app_streamlit.py
```

Open browser → `http://localhost:8501`

**Streamlit Features:**
- Upload images (multiple at once)
- Upload video with frame-by-frame prediction
- Realtime webcam (requires `streamlit-webrtc`)
- Fallback to snapshot if `streamlit-webrtc` not installed

---

### Step 3 — Deploy Python OpenCV (Realtime Desktop)

```bash
# Install dependencies
pip install opencv-python scikit-learn tensorflow scikit-image pillow

# Ensure ml_output folder exists in the same directory
# Run (interactive mode)
python app_realtime.py

# Or directly to specific mode:
python app_realtime.py --mode webcam
python app_realtime.py --mode video  --source path/to/video.mp4
python app_realtime.py --mode image  --source path/to/image.jpg
python app_realtime.py --mode webcam --camera 1  # second camera
```

**Keyboard controls (webcam/video mode):**

| Key       | Function             |
|-----------|----------------------|
| `Q` / `ESC` | Exit               |
| `S`       | Screenshot & save    |
| `P`       | Pause / Resume video |

---

## 📦 Training Output (ml_output/)

After training completes, this folder will contain:

```
ml_output/
├── metadata.json           ← algorithm info, classes, accuracy
├── class_names.json        ← list of class names
├── model_SVM.pkl           ← (Classical ML)
├── label_encoder.pkl       ← (Classical ML)
├── model_MobileNetV2.keras ← (Deep Learning)
├── best_model.keras        ← (Deep Learning, best checkpoint)
├── distribusi_kelas.png    ← dataset distribution chart
├── sample_gambar.png       ← image preview per class
├── confusion_matrix.png    ← model evaluation
└── training_history.png    ← (Deep Learning) loss/accuracy curves
```

---

## ⚙️ Customization

### Change Input Image Size
In `training_colab.ipynb` Cell 3:
```python
IMG_SIZE = (224, 224)  # larger = more accurate but slower
```

### Change Train/Test Split Ratio
```python
TEST_SIZE = 0.2  # 20% for test, 80% for training
```

### Change Realtime Prediction Frequency
In `app_realtime.py`:
```python
PREDICT_EVERY = 5  # predict every 5 frames (reduce for more frequent)
```

### Add New Classes
Simply add new folder in dataset, then retrain. No code changes needed.

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|--------|
| Model not found | Ensure `ml_output/` folder exists in the same directory as script |
| Camera won't open | Try `--camera 1` or `--camera 2` |
| `streamlit-webrtc` error | Install: `pip install streamlit-webrtc av` |
| DL model slow without GPU | Use MobileNetV2 or Classical ML for CPU |
| Low accuracy | Increase number of images per class (min 100+ per class) |
| Import error scikit-image | `pip install scikit-image` |

---

## 📋 Complete Requirements

```
# Core
numpy
opencv-python
scikit-learn
scikit-image
pillow
matplotlib
seaborn
tqdm

# Deep Learning (choose one)
tensorflow>=2.10.0

# Streamlit
streamlit>=1.28.0
streamlit-webrtc
av

# Optional
scipy
```
