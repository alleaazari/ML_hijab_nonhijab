"""
╔══════════════════════════════════════════════════════════════╗
║  ML Image Classification — Python Realtime (OpenCV)         ║
║  Input: Webcam realtime  |  Upload gambar  |  Upload video  ║
║  Cara run: python app_realtime.py                            ║
╚══════════════════════════════════════════════════════════════╝

REQUIREMENT:
    pip install opencv-python scikit-learn tensorflow
    pip install scikit-image pillow

KONTROL KEYBOARD (mode webcam/video):
    Q / ESC  → Keluar
    S        → Screenshot (simpan frame saat ini)
    P        → Pause / Resume (mode video)
    +/-      → Ubah ukuran window
"""

import os
import sys
import json
import pickle
import argparse
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
MODEL_DIR       = './ml_output'      # folder hasil training dari Colab
PREDICT_EVERY   = 5                  # prediksi setiap N frame (webcam/video)
WINDOW_WIDTH    = 960                # lebar window output
SCREENSHOT_DIR  = './screenshots'    # folder simpan screenshot
CONFIDENCE_HIGH = 0.70               # threshold warna hijau
CONFIDENCE_MID  = 0.40              # threshold warna kuning

# Warna overlay (BGR)
COLOR_HIGH   = (80, 220, 80)         # hijau
COLOR_MID    = (50, 200, 220)        # kuning
COLOR_LOW    = (60, 80, 220)         # merah
COLOR_BG     = (30, 30, 30)
COLOR_WHITE  = (240, 240, 240)
COLOR_GRAY   = (120, 120, 120)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
def load_model_and_meta(model_dir: str):
    meta_path = Path(model_dir) / 'metadata.json'
    if not meta_path.exists():
        nested_path = Path(model_dir) / 'ml_output'
        if (nested_path / 'metadata.json').exists():
            model_dir = str(nested_path)
            meta_path = nested_path / 'metadata.json'
        else:
            print(f'[ERROR] metadata.json tidak ditemukan di {model_dir}')
            sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    class_names = meta['class_names']
    is_dl       = meta['is_dl']
    algo        = meta['algorithm']

    if is_dl:
        import tensorflow as tf
        from tensorflow import keras

        model_path = Path(model_dir) / f'model_{algo}.keras'
        # fallback ke .h5 kalau .keras tidak ada
        if not model_path.exists():
            model_path_h5 = Path(model_dir) / f'model_{algo}.h5'
            if model_path_h5.exists():
                model_path = model_path_h5
            else:
                print(f'[ERROR] Model tidak ditemukan: {model_path}')
                sys.exit(1)

        print(f'[INFO] Loading DL model: {model_path}')
        print(f'[INFO] TensorFlow: {tf.__version__} | Keras: {keras.__version__}')

        # ── Strategy 1: load normal ──
        try:
            model = tf.keras.models.load_model(str(model_path))
            print('[INFO] Model loaded (strategy 1: default)')

        # ── Strategy 2: safe_mode=False (Keras 3.x) ──
        except Exception as e1:
            print(f'[WARN] Strategy 1 gagal: {type(e1).__name__}')
            try:
                model = tf.keras.models.load_model(
                    str(model_path), safe_mode=False
                )
                print('[INFO] Model loaded (strategy 2: safe_mode=False)')

            # ── Strategy 3: rebuild arsitektur + load weights ──
            except Exception as e2:
                print(f'[WARN] Strategy 2 gagal: {type(e2).__name__}')
                print('[INFO] Mencoba strategy 3: rebuild arsitektur + load weights...')
                try:
                    img_size   = tuple(meta['img_size'])
                    num_classes = meta['num_classes']

                    base_models = {
                        'MobileNetV2':    tf.keras.applications.MobileNetV2,
                        'ResNet50':       tf.keras.applications.ResNet50,
                        'EfficientNetB0': tf.keras.applications.EfficientNetB0,
                    }
                    if algo not in base_models:
                        raise ValueError(f'Algoritma {algo} tidak dikenal untuk rebuild')

                    BaseModel = base_models[algo]
                    base = BaseModel(
                        input_shape=(*img_size, 3),
                        include_top=False,
                        weights=None   # tanpa imagenet, nanti di-load dari weights
                    )
                    x = base.output
                    x = tf.keras.layers.GlobalAveragePooling2D()(x)
                    x = tf.keras.layers.Dense(256, activation='relu')(x)
                    x = tf.keras.layers.Dropout(0.4)(x)
                    x = tf.keras.layers.Dense(128, activation='relu')(x)
                    x = tf.keras.layers.Dropout(0.3)(x)
                    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
                    model = tf.keras.Model(inputs=base.input, outputs=outputs)

                    model.load_weights(str(model_path))
                    model.compile(
                        optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    print('[INFO] Model loaded (strategy 3: rebuild + load weights)')

                except Exception as e3:
                    print(f'[ERROR] Semua strategy gagal.\n')
                    print('=' * 60)
                    print('SOLUSI: Versi Keras/TF antara Colab dan lokal berbeda.')
                    print('Jalankan perintah ini di terminal:')
                    print()
                    print('  pip install "tensorflow==2.15.0" "keras==2.15.0"')
                    print()
                    print('ATAU export model ke .h5 di Colab dengan:')
                    print('  model.save("model_MobileNetV2.h5")')
                    print('=' * 60)
                    raise e3
    else:
        model_path = Path(model_dir) / f'model_{algo}.pkl'
        if not model_path.exists():
            print(f'[ERROR] Model tidak ditemukan: {model_path}')
            sys.exit(1)
        print(f'[INFO] Loading Classical ML model: {model_path}')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

    print(f'[INFO] Kelas: {class_names}')
    print(f'[INFO] Algoritma: {algo} | Akurasi: {meta["accuracy"]*100:.1f}%')
    return model, meta, class_names


# ─────────────────────────────────────────────
# FEATURE EXTRACTION & PREDICT
# ─────────────────────────────────────────────
def extract_hog(img_rgb: np.ndarray, target_size=(128, 128)) -> np.ndarray:
    from skimage.feature import hog
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, target_size)
    return hog(gray, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), visualize=False)


def filter_target_probs(all_probs: dict, focus_classes: list = None):
    """Filter probabilitas untuk fokus pada kelas target dan re-normalisasi."""
    if not focus_classes:
        pred_class, confidence = max(all_probs.items(), key=lambda x: x[1])
        return pred_class, confidence, all_probs

    matched = {}
    for fc in focus_classes:
        for k, v in all_probs.items():
            if k.lower() == fc.lower():
                matched[k] = v

    if not matched:
        pred_class, confidence = max(all_probs.items(), key=lambda x: x[1])
        return pred_class, confidence, all_probs

    total = sum(matched.values())
    if total > 0:
        norm_probs = {k: v / total for k, v in matched.items()}
    else:
        norm_probs = {k: 1.0 / len(matched) for k in matched}

    pred_class, confidence = max(norm_probs.items(), key=lambda x: x[1])
    return pred_class, confidence, norm_probs


def predict_frame(img_rgb: np.ndarray, model, meta: dict, class_names: list, focus_classes: list = None):
    """Return (pred_class, confidence, target_probs_dict)."""
    img_size = tuple(meta['img_size'])
    is_dl    = meta['is_dl']
    algo     = meta.get('algorithm', 'MobileNetV2')

    if is_dl:
        img = cv2.resize(img_rgb, img_size).astype('float32')
        if algo == 'MobileNetV2':
            import tensorflow as tf
            img_proc = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        elif algo == 'ResNet50':
            import tensorflow as tf
            img_proc = tf.keras.applications.resnet50.preprocess_input(img)
        elif algo == 'EfficientNetB0':
            import tensorflow as tf
            img_proc = tf.keras.applications.efficientnet.preprocess_input(img)
        else:
            img_proc = img / 255.0

        probs = model.predict(img_proc[np.newaxis], verbose=0)[0]
        all_probs = {cls: float(p) for cls, p in zip(class_names, probs)}
    else:
        feat = extract_hog(img_rgb, img_size)
        pred_raw = model.predict([feat])[0]
        pred_class_raw = class_names[pred_raw]
        if hasattr(model, 'predict_proba'):
            probs_arr  = model.predict_proba([feat])[0]
            all_probs  = {cls: float(p) for cls, p in zip(class_names, probs_arr)}
        else:
            all_probs  = {cls: (1.0 if cls == pred_class_raw else 0.0) for cls in class_names}

    pred_class, confidence, target_probs = filter_target_probs(all_probs, focus_classes)
    return pred_class, confidence, target_probs


# ─────────────────────────────────────────────
# RENDERING OVERLAY
# ─────────────────────────────────────────────
def get_color(confidence: float):
    if confidence >= CONFIDENCE_HIGH:
        return COLOR_HIGH
    elif confidence >= CONFIDENCE_MID:
        return COLOR_MID
    return COLOR_LOW


def render_overlay(frame: np.ndarray, pred_class: str, confidence: float,
                   all_probs: dict, class_names: list, fps: float = 0,
                   frame_idx: int = 0) -> np.ndarray:
    """Render hasil prediksi + bar chart ke dalam frame."""
    h, w = frame.shape[:2]
    out  = frame.copy()

    color = get_color(confidence)

    # ── Panel atas: prediksi utama ──
    panel_h = 90
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), COLOR_BG, -1)
    out = cv2.addWeighted(overlay, 0.75, out, 0.25, 0)

    label = f'{pred_class.upper()}   {confidence*100:.1f}%'
    cv2.putText(out, label, (15, 55),
                cv2.FONT_HERSHEY_DUPLEX, 1.3, color, 2, cv2.LINE_AA)

    # Confidence bar
    bar_w = int((w - 30) * confidence)
    cv2.rectangle(out, (15, 68), (w - 15, 82), (50, 50, 50), -1)
    cv2.rectangle(out, (15, 68), (15 + bar_w, 82), color, -1)

    # ── Panel kanan: semua kelas ──
    panel_right_w = 220
    max_h_avail   = h - panel_h - 50
    row_h         = min(36, max(24, max_h_avail // max(1, len(class_names))))
    panel_right_h = 30 + len(class_names) * row_h
    px, py = w - panel_right_w - 10, panel_h + 10

    overlay2 = out.copy()
    cv2.rectangle(overlay2, (px, py), (px + panel_right_w, py + panel_right_h),
                  COLOR_BG, -1)
    out = cv2.addWeighted(overlay2, 0.72, out, 0.28, 0)

    cv2.putText(out, 'PROBABILITAS', (px + 8, py + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GRAY, 1)

    bar_h  = max(10, row_h - 18)
    font_s = 0.38 if row_h >= 30 else 0.32
    for i, cls in enumerate(sorted(all_probs.items(), key=lambda x: -x[1])):
        cls_name, prob = cls
        cy = py + 28 + i * row_h
        bar_len = int((panel_right_w - 20) * prob)
        c = get_color(prob)

        cv2.rectangle(out, (px + 8, cy), (px + 8 + bar_len, cy + bar_h), c, -1)
        cv2.rectangle(out, (px + 8, cy), (px + panel_right_w - 12, cy + bar_h),
                      COLOR_GRAY, 1)

        label_txt = f'{cls_name[:15]}: {prob*100:.0f}%'
        cv2.putText(out, label_txt, (px + 10, cy + bar_h + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, font_s,
                    COLOR_WHITE if prob > 0.1 else COLOR_GRAY, 1)

    # ── Panel bawah: info ──
    info_y = h - 28
    overlay3 = out.copy()
    cv2.rectangle(overlay3, (0, h - 38), (w, h), COLOR_BG, -1)
    out = cv2.addWeighted(overlay3, 0.7, out, 0.3, 0)

    info = f'FPS: {fps:.1f}  |  Frame: {frame_idx}  |  [Q] Keluar  [S] Screenshot  [P] Pause'
    cv2.putText(out, info, (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_GRAY, 1, cv2.LINE_AA)

    return out


def resize_frame(frame: np.ndarray, target_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = target_width / w
    return cv2.resize(frame, (target_width, int(h * scale)))


def save_screenshot(frame: np.ndarray, save_dir: str, pred_class: str):
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(save_dir, f'screenshot_{pred_class}_{ts}.jpg')
    cv2.imwrite(path, frame)
    print(f'[INFO] Screenshot tersimpan: {path}')
    return path


# ─────────────────────────────────────────────
# MODE 1: WEBCAM REALTIME
# ─────────────────────────────────────────────
def run_webcam(model, meta, class_names, camera_index=0):
    print('\n[WEBCAM] Membuka kamera...')
    # Gunakan DirectShow (CAP_DSHOW) di Windows agar webcam USB terbuka lebih cepat dan tidak hang
    if os.name == 'nt':
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)
    else:
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print('[ERROR] Kamera tidak bisa dibuka!')
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow('ML Classifier — Webcam', cv2.WINDOW_NORMAL)

    pred_class  = 'Menunggu...'
    confidence  = 0.0
    all_probs   = {cls: 0.0 for cls in class_names}
    frame_idx   = 0
    prev_time   = cv2.getTickCount()
    fps         = 0.0

    print('[INFO] Tekan Q/ESC untuk keluar, S untuk screenshot')

    while True:
        ret, frame = cap.read()
        if not ret:
            print('[ERROR] Gagal baca frame kamera')
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Prediksi setiap N frame
        if frame_idx % PREDICT_EVERY == 0:
            pred_class, confidence, all_probs = predict_frame(
                frame_rgb, model, meta, class_names, focus_classes=['hijab', 'nonhijab']
            )

        # Hitung FPS
        curr_time = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (curr_time - prev_time)
        prev_time = curr_time

        # Render
        frame_out = render_overlay(
            frame_rgb, pred_class, confidence, all_probs,
            class_names, fps, frame_idx
        )
        frame_bgr = cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR)
        frame_bgr = resize_frame(frame_bgr, WINDOW_WIDTH)
        cv2.imshow('ML Classifier — Webcam', frame_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # Q atau ESC
            break
        elif key == ord('s'):
            save_screenshot(frame_bgr, SCREENSHOT_DIR, pred_class)

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print('[INFO] Webcam ditutup.')


# ─────────────────────────────────────────────
# MODE 2: VIDEO FILE
# ─────────────────────────────────────────────
def run_video(video_path: str, model, meta, class_names):
    print(f'\n[VIDEO] Membuka: {video_path}')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'[ERROR] Video tidak bisa dibuka: {video_path}')
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_fps     = cap.get(cv2.CAP_PROP_FPS)
    print(f'[INFO] Total frame: {total_frames} | FPS video: {orig_fps:.1f}')

    cv2.namedWindow('ML Classifier — Video', cv2.WINDOW_NORMAL)

    pred_class = 'Loading...'
    confidence = 0.0
    all_probs  = {cls: 0.0 for cls in class_names}
    frame_idx  = 0
    paused     = False
    prev_time  = cv2.getTickCount()
    fps        = 0.0

    print('[INFO] Q/ESC: Keluar | S: Screenshot | P: Pause/Resume')

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print('[INFO] Video selesai.')
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if frame_idx % PREDICT_EVERY == 0:
                pred_class, confidence, all_probs = predict_frame(
                    frame_rgb, model, meta, class_names, focus_classes=['hijab', 'nonhijab']
                )

            curr_time = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (curr_time - prev_time)
            prev_time = curr_time

            frame_out = render_overlay(
                frame_rgb, pred_class, confidence, all_probs,
                class_names, fps, frame_idx
            )

            # Progress bar bawah
            h, w = frame_out.shape[:2]
            progress = frame_idx / max(total_frames, 1)
            prog_w   = int(w * progress)
            cv2.rectangle(frame_out, (0, h - 6), (w, h - 2), (50, 50, 50), -1)
            cv2.rectangle(frame_out, (0, h - 6), (prog_w, h - 2), COLOR_HIGH, -1)

            frame_bgr = cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR)
            frame_bgr = resize_frame(frame_bgr, WINDOW_WIDTH)
            cv2.imshow('ML Classifier — Video', frame_bgr)

            frame_idx += 1

            if frame_idx % 30 == 0:
                pct = frame_idx / total_frames * 100
                print(f'  Progress: {pct:.0f}% ({frame_idx}/{total_frames}) — {pred_class} {confidence*100:.1f}%')

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s') and not paused:
            save_screenshot(frame_bgr, SCREENSHOT_DIR, pred_class)
        elif key == ord('p'):
            paused = not paused
            status = '⏸ PAUSED' if paused else '▶ RESUMED'
            print(f'[INFO] {status}')

    cap.release()
    cv2.destroyAllWindows()
    print('[INFO] Video selesai diproses.')


# ─────────────────────────────────────────────
# MODE 3: GAMBAR STATIS
# ─────────────────────────────────────────────
def run_image(image_path: str, model, meta, class_names):
    print(f'\n[IMAGE] Memproses: {image_path}')
    img = cv2.imread(image_path)
    if img is None:
        print(f'[ERROR] Gambar tidak bisa dibaca: {image_path}')
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pred_class, confidence, all_probs = predict_frame(img_rgb, model, meta, class_names, focus_classes=['hijab', 'nonhijab'])

    print(f'\n  🎯 Prediksi  : {pred_class}')
    print(f'  📊 Confidence: {confidence*100:.2f}%')
    print('\n  Semua Kelas:')
    for cls, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
        bar = '█' * int(prob * 25)
        print(f'    {cls:20s}: {bar} {prob*100:.1f}%')

    frame_out = render_overlay(img_rgb, pred_class, confidence, all_probs, class_names)
    frame_bgr = cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR)
    frame_bgr = resize_frame(frame_bgr, WINDOW_WIDTH)

    cv2.namedWindow('ML Classifier — Image', cv2.WINDOW_NORMAL)
    cv2.imshow('ML Classifier — Image', frame_bgr)
    print('\n[INFO] Tekan Q/ESC untuk keluar, S untuk screenshot')

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            save_screenshot(frame_bgr, SCREENSHOT_DIR, pred_class)

    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# MENU INTERAKTIF
# ─────────────────────────────────────────────
def interactive_menu(model, meta, class_names):
    print('\n' + '='*55)
    print('  🤖 ML Image Classifier — Realtime')
    print(f'  Algoritma : {meta["algorithm"]}')
    print(f'  Kelas     : {", ".join(class_names)}')
    print(f'  Akurasi   : {meta["accuracy"]*100:.1f}%')
    print('='*55)
    print('\nPilih mode:')
    print('  1. 📷 Webcam Realtime')
    print('  2. 🎥 Video File')
    print('  3. 🖼️  Gambar / Foto')
    print('  0. ❌ Keluar\n')

    choice = input('Masukkan pilihan (0-3): ').strip()

    if choice == '1':
        cam_idx = input('Index kamera (default 0): ').strip()
        cam_idx = int(cam_idx) if cam_idx.isdigit() else 0
        run_webcam(model, meta, class_names, cam_idx)

    elif choice == '2':
        vpath = input('Path video: ').strip().strip('"\'')
        if not os.path.exists(vpath):
            print(f'[ERROR] File tidak ditemukan: {vpath}')
        else:
            run_video(vpath, model, meta, class_names)

    elif choice == '3':
        ipath = input('Path gambar: ').strip().strip('"\'')
        if not os.path.exists(ipath):
            print(f'[ERROR] File tidak ditemukan: {ipath}')
        else:
            run_image(ipath, model, meta, class_names)

    elif choice == '0':
        print('Sampai jumpa! 👋')
        sys.exit(0)
    else:
        print('[ERROR] Pilihan tidak valid.')


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML Realtime Classifier')
    parser.add_argument('--model-dir', default=MODEL_DIR, help='Path folder model')
    parser.add_argument('--mode',      choices=['webcam', 'video', 'image', 'menu'],
                        default='menu', help='Mode input')
    parser.add_argument('--source',    default=None,
                        help='Path file untuk mode video/image')
    parser.add_argument('--camera',    type=int, default=0,
                        help='Index kamera (default: 0)')
    args = parser.parse_args()

    # Load model
    model, meta, class_names = load_model_and_meta(args.model_dir)

    # Jalankan sesuai mode
    if args.mode == 'menu':
        while True:
            interactive_menu(model, meta, class_names)
            again = input('\nCoba mode lain? (y/n): ').strip().lower()
            if again != 'y':
                break

    elif args.mode == 'webcam':
        run_webcam(model, meta, class_names, args.camera)

    elif args.mode == 'video':
        if not args.source:
            print('[ERROR] --source diperlukan untuk mode video')
        else:
            run_video(args.source, model, meta, class_names)

    elif args.mode == 'image':
        if not args.source:
            print('[ERROR] --source diperlukan untuk mode image')
        else:
            run_image(args.source, model, meta, class_names)
