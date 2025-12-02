import streamlit as st
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
import base64
import traceback

st.set_page_config(layout="wide", page_title="DERMALYTICS")

# ==================== CONFIG ====================
CLASS_NAMES = {
    0: "akiec", 1: "bcc", 2: "bkl", 3: "df",
    4: "mel", 5: "nv", 6: "vasc"
}

MODEL_PATHS = {
    "Model 1 (Recommended)"      : r"C:\Users\lenovo\Documents\CAPSTONE\CAPSTONE C\SourceCode\models\best_model.h5",
    "Model 2 (InceptionV3)"      : r"C:\Users\lenovo\Documents\CAPSTONE\CAPSTONE C\SourceCode\models\InceptionV3_HAM10000.h5",
    "Model 3 (MobileNetV2)"      : r"C:\Users\lenovo\Documents\CAPSTONE\CAPSTONE C\SourceCode\models\MobileNetV2_HAM10000.h5",
    "Model 4 (DenseNet121)"      : r"C:\Users\lenovo\Documents\CAPSTONE\CAPSTONE C\SourceCode\models\DenseNet121_HAM10000.h5",
    "Model 5 (Xception)"         : r"C:\Users\lenovo\Documents\CAPSTONE\CAPSTONE C\SourceCode\models\Xception_HAM10000.h5",
    "Model 6 (EfficientNetB0)"   : r"C:\Users\lenovo\Documents\CAPSTONE\CAPSTONE C\SourceCode\models\EfficientNetB0_HAM10000.h5",
    "Model 7 (EfficientNetB3)"   : r"C:\Users\lenovo\Documents\CAPSTONE\CAPSTONE C\SourceCode\models\EfficientNetB3_HAM10000.h5",
    "Model 8 (ResNet101)"        : r"C:\Users\lenovo\Documents\CAPSTONE\CAPSTONE C\SourceCode\models\ResNet101_HAM10000.h5",
    "Model 9 (ResNet50)"         : r"C:\Users\lenovo\Documents\CAPSTONE\CAPSTONE C\SourceCode\models\ResNet50_HAM10000.h5"
}

# ==================== LOAD MODELS ====================
@st.cache_resource
def load_all_models():
    models = {}
    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            with st.spinner(f"Loading {name} ..."):
                try:
                    model = load_model(path, compile=False)
                    models[name] = model
                    st.success(f"{name} loaded")
                except:
                    st.error(f"Failed to load {name}")
        else:
            st.warning(f"Model file missing: {path}")
    return models

models = load_all_models()

# ==================== PREPROCESS ====================
def preprocess_image_pil(img):
    img = img.resize((224, 224))
    arr = image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0), np.array(img)


# ==================== GRAD-CAM++ ====================
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            return layer.name
    raise ValueError("No conv layer found")

def get_gradcam_plus_plus(model, img_array):
    try:
        target_layer_name = find_last_conv_layer(model)
        target_layer = model.get_layer(target_layer_name)
        model_output = model.output if not isinstance(model.output, list) else model.output[0]

        grad_model = tf.keras.models.Model(model.inputs, [target_layer.output, model_output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        grads = tf.nn.relu(grads)

        weights = tf.reduce_mean(grads, axis=(1, 2))
        cam = tf.reduce_sum(weights[:, None, None, :] * conv_outputs, axis=-1)
        cam = tf.nn.relu(cam)
        cam = cam.numpy()[0]

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = np.uint8(255 * cam)
        cam = cv2.resize(cam, (224, 224))
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        return heatmap, predictions.numpy()[0], class_idx
    except Exception as e:
        st.error("Grad-CAM++ failed")
        st.write(str(e))
        traceback.print_exc()
        dummy = np.zeros((224, 224), dtype=np.uint8)
        return cv2.applyColorMap(dummy, cv2.COLORMAP_JET), np.zeros(7), 0


# ==================== UI ====================
st.title("🩺 DERMALYTICS – Skin Lesion Classifier")
st.write("Upload an image and let AI detect skin lesion type with heatmap explanation.")

model_name = st.selectbox("Select a model:", list(models.keys()))

uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "png", "jpeg"])

if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")
    st.image(pil_img, caption="Uploaded Image")

    img_array, display_img = preprocess_image_pil(pil_img)

    model = models[model_name]

    with st.spinner("Running prediction..."):
        preds = model.predict(img_array)[0]
        class_id = np.argmax(preds)
        confidence = preds[class_id]

        heatmap, cam_probs, cam_class = get_gradcam_plus_plus(model, img_array)

        overlay = cv2.addWeighted(cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR), 0.5, heatmap, 0.5, 0)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔥 Grad-CAM++ Heatmap")
        st.image(heatmap, channels="BGR")

        st.subheader("🔍 Overlay")
        st.image(overlay_rgb, channels="RGB")

    with col2:
        st.subheader("Prediction")
        st.write(f"**Class:** {CLASS_NAMES[class_id]}")  
        st.write(f"**Confidence:** {confidence * 100:.2f}%")

        st.write("---")
        st.subheader("All Probabilities:")
        for i, p in enumerate(preds):
            st.write(f"{CLASS_NAMES[i]} → {p*100:.2f}%")

