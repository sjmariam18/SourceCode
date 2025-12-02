import streamlit as st
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2 as cv

# Critical fix for Streamlit Cloud + OpenCV
os.environ["OPENCV_VIDEOIO_MSP_BACKENDS"] = "ANY"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "dummy"

st.set_page_config(layout="wide", page_title="DERMALYTICS - Skin Lesion AI")

# ==================== CONFIG ====================
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

MODEL_PATHS = {
    "Model 1 (Recommended)"       : "models/best_model.h5",
    "InceptionV3"                 : "models/InceptionV3_HAM10000.h5",
    "MobileNetV2"                 : "models/MobileNetV2_HAM10000.h5",
    "DenseNet121"                 : "models/DenseNet121_HAM10000.h5",
    "Xception"                    : "models/Xception_HAM10000.h5",
    "EfficientNetB0"              : "models/EfficientNetB0_HAM10000.h5",
    "EfficientNetB3"              : "models/EfficientNetB3_HAM10000.h5",
    "ResNet101"                   : "models/ResNet101_HAM10000.h5",
    "ResNet50"                    : "models/ResNet50_HAM10000.h5",
}

# ==================== LOAD MODELS ====================
@st.cache_resource
def load_models():
    models = {}
    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            with st.spinner(f"Loading {name}..."):
                try:
                    model = load_model(path, compile=False)
                    models[name] = model
                    st.success(f"✓ {name}")
                except Exception as e:
                    st.error(f"✗ {name}: {e}")
        else:
            st.warning(f"⚠ {name} → {path} not found")
    return models

models = load_models()

# ==================== PREPROCESS & GRADCAM ====================
def preprocess(img):
    img = img.resize((224, 224))
    arr = image.img_to_array(img) / 255.0
    return np.expand_dims(arr, 0), np.array(img)

def get_gradcam(model, img_array):
    try:
        # Find last conv layer
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:
                last_conv = layer
                break
        else:
            return np.zeros((224,224,3), np.uint8)

        grad_model = tf.keras.models.Model(
            model.inputs, [last_conv.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_array)
            class_idx = tf.argmax(preds[0])
            loss = preds[:, class_idx]

        grads = tape.gradient(loss, conv_out)[0]
        weights = tf.reduce_mean(grads, axis=(0, 1))
        cam = tf.reduce_sum(weights * conv_out[0], axis=-1)
        cam = np.maximum(cam, 0)
        cam = cv.resize(cam.numpy(), (224, 224))
        cam = cam / (cam.max() + 1e-8)
        cam = np.uint8(255 * cam)
        heatmap = cv.applyColorMap(cam, cv.COLORMAP_JET)
        return heatmap
    except:
        return np.zeros((224,224,3), np.uint8)

# ==================== UI ====================
st.title("DERMALYTICS – AI Skin Lesion Classifier")
st.markdown("Upload a dermatoscopic image → get diagnosis + Grad-CAM++ explanation")

if not models:
    st.error("No models found! Make sure your `.h5` files are in the `models/` folder.")
    st.stop()

selected_model = st.selectbox("Choose Model", list(models.keys()))
model = models[selected_model]

uploaded = st.file_uploader("Upload skin lesion image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)

    arr, display_img = preprocess(img)

    with st.spinner("Analyzing..."):
        pred = model.predict(arr)[0]
        idx = np.argmax(pred)
        confidence = pred[idx]

        heatmap = get_gradcam(model, arr)
        overlay = cv.addWeighted(display_img, 0.6, heatmap, 0.4, 0)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Grad-CAM++ Heatmap")
        st.image(heatmap, channels="BGR", use_column_width=True)
        st.subheader("Overlay")
        st.image(overlay, use_column_width=True)

    with col2:
        st.subheader("Prediction")
        st.metric("Diagnosis", CLASS_NAMES[idx].upper(), f"{confidence:.1%}")

        st.bar_chart({CLASS_NAMES[i]: float(pred[i]) for i in range(7)})