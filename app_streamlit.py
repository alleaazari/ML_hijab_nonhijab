"""
╔══════════════════════════════════════════════════════════════╗
║  ML Image Classification — Streamlit App (Realtime)         ║
║  Input: Webcam realtime  |  Upload gambar/video             ║
║  Cara run: streamlit run app_streamlit.py                    ║
╚══════════════════════════════════════════════════════════════╝

REQUIREMENT:
    pip install streamlit opencv-python scikit-learn tensorflow
    pip install scikit-image streamlit-webrtc av pillow

STRUKTUR FILE:
    ml_output/
    ├── metadata.json
    ├── class_names.json
    ├── model_SVM.pkl          ← jika classical ML
    ├── label_encoder.pkl      ← jika classical ML
    └── model_MobileNetV2.keras ← jika deep learning
"""

import os
import json
import pickle
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from pathlib import Path

# ─────────────────────────────────────────────
# KONFIGURASI — sesuaikan path model
# ─────────────────────────────────────────────
MODEL_DIR = './ml_output'   # folder hasil training dari Colab


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model_and_meta():
    meta_path = Path(MODEL_DIR) / 'metadata.json'
    if not meta_path.exists():
        return None, None, None

    with open(meta_path) as f:
        meta = json.load(f)

    class_names = meta['class_names']
    img_size    = tuple(meta['img_size'])
    is_dl       = meta['is_dl']
    algo        = meta['algorithm']

    if is_dl:
        import tensorflow as tf

        # Cek .keras dulu, fallback ke .h5
        model_path = Path(MODEL_DIR) / f'model_{algo}.keras'
        if not model_path.exists():
            model_path = Path(MODEL_DIR) / f'model_{algo}.h5'
        if not model_path.exists():
            return None, None, None

        # Strategy 1: load normal
        try:
            model = tf.keras.models.load_model(str(model_path))

        # Strategy 2: safe_mode=False (Keras 3.x)
        except Exception:
            try:
                model = tf.keras.models.load_model(str(model_path), safe_mode=False)

            # Strategy 3: rebuild arsitektur + load weights
            except Exception:
                img_size_tuple = tuple(meta['img_size'])
                num_classes    = meta['num_classes']

                base_models = {
                    'MobileNetV2':    tf.keras.applications.MobileNetV2,
                    'ResNet50':       tf.keras.applications.ResNet50,
                    'EfficientNetB0': tf.keras.applications.EfficientNetB0,
                }
                BaseModel = base_models.get(algo)
                if BaseModel is None:
                    return None, None, None

                base = BaseModel(
                    input_shape=(*img_size_tuple, 3),
                    include_top=False,
                    weights=None
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
    else:
        model_path = Path(MODEL_DIR) / f'model_{algo}.pkl'
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

    return model, meta, class_names


def preprocess_image(img_rgb: np.ndarray, img_size: tuple, is_dl: bool):
    """Resize + normalize gambar untuk inferensi."""
    img = cv2.resize(img_rgb, img_size)
    if is_dl:
        return img.astype('float32') / 255.0
    else:
        return img


def extract_features_classical(img_rgb: np.ndarray):
    """HOG feature extraction untuk Classical ML."""
    from skimage.feature import hog
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (128, 128))
    feat = hog(gray, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), visualize=False)
    return feat


def predict(img_rgb: np.ndarray, model, meta: dict, class_names: list):
    """Jalankan prediksi dan return class + probabilities."""
    img_size = tuple(meta['img_size'])
    is_dl    = meta['is_dl']

    if is_dl:
        img_proc = preprocess_image(img_rgb, img_size, is_dl=True)
        probs = model.predict(img_proc[np.newaxis], verbose=0)[0]
        pred_idx   = int(np.argmax(probs))
        pred_class = class_names[pred_idx]
        confidence = float(probs[pred_idx])
        all_probs  = {cls: float(p) for cls, p in zip(class_names, probs)}
    else:
        feat = extract_features_classical(img_rgb)
        pred_raw = model.predict([feat])[0]
        pred_class = class_names[pred_raw]

        if hasattr(model, 'predict_proba'):
            probs_arr = model.predict_proba([feat])[0]
            confidence = float(np.max(probs_arr))
            all_probs  = {cls: float(p) for cls, p in zip(class_names, probs_arr)}
        else:
            confidence = 1.0
            all_probs  = {cls: (1.0 if cls == pred_class else 0.0) for cls in class_names}

    return pred_class, confidence, all_probs


def draw_overlay(frame: np.ndarray, pred_class: str,
                 confidence: float, class_names: list) -> np.ndarray:
    """Gambar overlay hasil prediksi pada frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Background box
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Teks prediksi
    label = f'{pred_class}  {confidence*100:.1f}%'
    cv2.putText(frame, label, (15, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 100), 2, cv2.LINE_AA)

    # Confidence bar
    bar_w = int((w - 30) * confidence)
    cv2.rectangle(frame, (15, 60), (w - 15, 72), (60, 60, 60), -1)
    cv2.rectangle(frame, (15, 60), (15 + bar_w, 72), (0, 255, 100), -1)

    return frame


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title='ML Image Classifier',
        page_icon='🤖',
        layout='wide'
    )

    st.title('🤖 ML Image Classifier')
    st.markdown('Klasifikasi gambar otomatis menggunakan model yang sudah ditraining.')

    # Load model
    with st.spinner('Loading model...'):
        model, meta, class_names = load_model_and_meta()

    if model is None:
        st.error(f'❌ Model tidak ditemukan di `{MODEL_DIR}/`')
        st.info('Pastikan sudah training di Colab dan copy folder `ml_output` ke direktori ini.')
        return

    # Info model
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Algoritma', meta['algorithm'])
    col2.metric('Jumlah Kelas', meta['num_classes'])
    col3.metric('Akurasi Training', f"{meta['accuracy']*100:.1f}%")
    col4.metric('Input Size', f"{meta['img_size'][0]}×{meta['img_size'][1]}")

    st.divider()

    # Pilih mode input
    mode = st.radio(
        '📥 Pilih Mode Input',
        ['📁 Upload Gambar', '🎥 Upload Video', '📷 Webcam Realtime'],
        horizontal=True
    )

    # ── MODE 1: Upload Gambar ──
    if mode == '📁 Upload Gambar':
        uploaded = st.file_uploader(
            'Upload gambar (JPG, PNG, WEBP)',
            type=['jpg', 'jpeg', 'png', 'webp', 'bmp'],
            accept_multiple_files=True
        )

        if uploaded:
            for upf in uploaded:
                img_pil = Image.open(upf).convert('RGB')
                img_arr = np.array(img_pil)

                pred_class, confidence, all_probs = predict(
                    img_arr, model, meta, class_names
                )

                c1, c2 = st.columns([1, 1])
                with c1:
                    st.image(img_pil, caption=upf.name, use_column_width=True)
                with c2:
                    st.subheader('Hasil Prediksi')

                    color = 'green' if confidence >= 0.7 else 'orange' if confidence >= 0.4 else 'red'
                    st.markdown(
                        f'<h2 style="color:{color}">'
                        f'🏷️ {pred_class}</h2>',
                        unsafe_allow_html=True
                    )
                    st.metric('Confidence', f'{confidence*100:.1f}%')
                    st.markdown('**Semua Kelas:**')
                    for cls, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
                        st.progress(prob, text=f'{cls}: {prob*100:.1f}%')
                st.divider()

    # ── MODE 2: Upload Video ──
    elif mode == '🎥 Upload Video':
        uploaded_video = st.file_uploader(
            'Upload video (MP4, AVI, MOV)',
            type=['mp4', 'avi', 'mov', 'mkv']
        )

        if uploaded_video:
            # Simpan sementara
            tmp_path = '/tmp/uploaded_video.mp4'
            with open(tmp_path, 'wb') as f:
                f.write(uploaded_video.read())

            skip_frame = st.slider('Prediksi setiap N frame', 1, 30, 5)
            start_btn  = st.button('▶️ Mulai Proses Video')

            if start_btn:
                cap = cv2.VideoCapture(tmp_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                st.info(f'Total frame: {frame_count} | FPS: {fps:.0f}')

                frame_placeholder = st.empty()
                info_placeholder  = st.empty()
                progress_bar      = st.progress(0)

                frame_idx = 0
                pred_class, confidence, all_probs = '...', 0.0, {}

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    if frame_idx % skip_frame == 0:
                        pred_class, confidence, all_probs = predict(
                            frame_rgb, model, meta, class_names
                        )

                    # Overlay
                    frame_out = draw_overlay(frame_rgb.copy(), pred_class, confidence, class_names)
                    frame_placeholder.image(frame_out, channels='RGB', use_column_width=True)

                    info_placeholder.markdown(
                        f'**Frame {frame_idx}/{frame_count}** — '
                        f'Prediksi: **{pred_class}** ({confidence*100:.1f}%)'
                    )
                    progress_bar.progress(min(frame_idx / frame_count, 1.0))
                    frame_idx += 1

                cap.release()
                st.success('✅ Video selesai diproses!')

    # ── MODE 3: Webcam Realtime ──
    elif mode == '📷 Webcam Realtime':
        st.info(
            '📌 **Cara pakai Webcam:**  \n'
            '1. Klik tombol "Aktifkan Webcam" di bawah  \n'
            '2. Izinkan akses kamera di browser  \n'
            '3. Klik "📸 Capture & Prediksi" untuk mengambil gambar'
        )

        # Coba pakai streamlit-webrtc untuk realtime
        try:
            from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
            import av

            class RealtimeClassifier(VideoTransformerBase):
                def __init__(self):
                    self.model      = model
                    self.meta       = meta
                    self.class_names = class_names
                    self.pred_class  = ''
                    self.confidence  = 0.0
                    self.frame_count = 0
                    self.skip        = 5  # prediksi setiap 5 frame

                def recv(self, frame):
                    img = frame.to_ndarray(format='bgr24')
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    self.frame_count += 1
                    if self.frame_count % self.skip == 0:
                        self.pred_class, self.confidence, _ = predict(
                            img_rgb, self.model, self.meta, self.class_names
                        )

                    if self.pred_class:
                        img_out = draw_overlay(
                            img_rgb.copy(),
                            self.pred_class,
                            self.confidence,
                            self.class_names
                        )
                        img_bgr = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
                    else:
                        img_bgr = img

                    return av.VideoFrame.from_ndarray(img_bgr, format='bgr24')

            webrtc_streamer(
                key='realtime-classifier',
                video_transformer_factory=RealtimeClassifier,
                media_stream_constraints={'video': True, 'audio': False},
                async_transform=True,
            )

        except ImportError:
            # Fallback: snapshot webcam via st.camera_input
            st.warning(
                '`streamlit-webrtc` tidak terinstall. '
                'Menggunakan mode snapshot (install `pip install streamlit-webrtc av` untuk realtime penuh).'
            )

            img_file = st.camera_input('📷 Ambil Foto')
            if img_file:
                img_pil = Image.open(img_file).convert('RGB')
                img_arr = np.array(img_pil)

                pred_class, confidence, all_probs = predict(
                    img_arr, model, meta, class_names
                )

                c1, c2 = st.columns([1, 1])
                with c1:
                    st.image(img_pil, use_column_width=True)
                with c2:
                    st.subheader('Hasil Prediksi')
                    color = 'green' if confidence >= 0.7 else 'orange' if confidence >= 0.4 else 'red'
                    st.markdown(
                        f'<h2 style="color:{color}">🏷️ {pred_class}</h2>',
                        unsafe_allow_html=True
                    )
                    st.metric('Confidence', f'{confidence*100:.1f}%')
                    for cls, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
                        st.progress(prob, text=f'{cls}: {prob*100:.1f}%')

    # Sidebar: info kelas
    with st.sidebar:
        st.header('📋 Info Model')
        st.write(f'**Algoritma:** {meta["algorithm"]}')
        st.write(f'**Mode:** {"Deep Learning" if meta["is_dl"] else "Classical ML"}')
        st.write(f'**Akurasi:** {meta["accuracy"]*100:.1f}%')
        st.write(f'**Input Size:** {meta["img_size"][0]}×{meta["img_size"][1]}')
        st.divider()
        st.write('**Kelas yang Dikenali:**')
        for i, cls in enumerate(class_names):
            st.write(f'  {i+1}. {cls}')


if __name__ == '__main__':
    main()