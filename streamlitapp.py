# import streamlit as st
# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from PIL import Image
# import cv2 as cv

# # ==================== PAGE CONFIG & STYLE ====================
# st.set_page_config(
#     page_title="DERMALYTICS – AI-Powered Skin Lesion Analysis",
#     page_icon="🔬",
#     layout="centered",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for professional medical-tech aesthetic
# st.markdown("""
# <style>
#     .main {
#         background-color: #f8f9fc;
#     }
#     .header-title {
#         font-size: 3.2rem;
#         font-weight: 700;
#         background: linear-gradient(90deg, #1e3a8a, #2c7be5);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         text-align: center;
#         margin-bottom: 10px;
#     }
#     .subtitle {
#         font-size: 1.3rem;
#         color: #555;
#         text-align: center;
#         margin-bottom: 40px;
#         font-weight: 300;
#     }
#     .stSelectbox > div > div > select {
#         border-radius: 12px;
#         border: 1px solid #2c7be5;
#     }
#     .metric-card {
#         background: white;
#         padding: 20px;
#         border-radius: 16px;
#         box-shadow: 0 6px 20px rgba(0,0,0,0.08);
#         border-left: 6px solid #2c7be5;
#     }
#     .heatmap {
#         border-radius: 16px;
#         box-shadow: 0 8px 25px rgba(0,0,0,0.1);
#     }
#     .stButton>button {
#         background: linear-gradient(90deg, #1e3a8a, #2c7be5);
#         color: white;
#         border-radius: 12px;
#         height: 50px;
#         font-weight: bold;
#         border: none;
#         transition: all 0.3s;
#     }
#     .stButton>button:hover {
#         transform: translateY(-3px);
#         box-shadow: 0 10px 20px rgba(44,123,229,0.3);
#     }
#     .footer {
#         text-align: center;
#         margin-top: 80px;
#         color: #888;
#         font-size: 0.9rem;
#     }
#     .warning-box {
#         padding: 15px;
#         background-color: #fff3cd;
#         border-left: 6px solid #ffc107;
#         border-radius: 8px;
#         margin: 20px 0;
#     }
# </style>
# """, unsafe_allow_html=True)

# # ==================== CONFIG ====================
# CLASS_NAMES = {
#     "akiec": "Actinic Keratoses / Bowen's Disease",
#     "bcc": "Basal Cell Carcinoma",
#     "bkl": "Benign Keratosis-like Lesions",
#     "df": "Dermatofibroma",
#     "mel": "Melanoma",
#     "nv": "Melanocytic Nevi",
#     "vasc": "Vascular Lesions"
# }

# SHORT_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# MODEL_PATHS = {
#     "Model 1 (Recommended)" : "models/best_model.h5",
#     "InceptionV3": "models/InceptionV3_HAM10000.h5",
#     "MobileNetV2": "models/MobileNetV2_HAM10000.h5",
#     "DenseNet121": "models/DenseNet121_HAM10000.h5",
#     "Xception": "models/Xception_HAM10000.h5",
#     "EfficientNetB0": "models/EfficientNetB0_HAM10000.h5",
#     "EfficientNetB3": "models/EfficientNetB3_HAM10000.h5",
#     "ResNet101": "models/ResNet101_HAM10000.h5",
#     "ResNet50": "models/ResNet50_HAM10000.h5"
# }

# MODEL_PATHS = {
#     "Model 1 (Recommended)" : "models/best_model.h5",
#     "InceptionV3": "models/InceptionV3_HAM10000.h5",
#     "MobileNetV2": "models/MobileNetV2_HAM10000.h5",
#     "DenseNet121": "models/DenseNet121_HAM10000.h5",
#     "Xception": "models/Xception_HAM10000.h5",
#     "EfficientNetB0": "models/EfficientNetB0_HAM10000.h5",
#     "EfficientNetB3": "models/EfficientNetB3_HAM10000.h5",
#     "ResNet101": "models/ResNet101_HAM10000.h5",
#     "ResNet50": "models/ResNet50_HAM10000.h5"
# }

# # ==================== LOAD MODELS ====================
# @st.cache_resource(show_spinner=False)
# def load_models():
#     models = {}
#     with st.spinner("Initializing AI models... This may take a moment."):
#         for name, path in MODEL_PATHS.items():
#             if os.path.exists(path):
#                 try:
#                     model = load_model(path, compile=False)
#                     models[name] = model
#                 except Exception as e:
#                     st.error(f"Failed to load {name}: {e}")
#             else:
#                 st.warning(f"Model not found: `{path}`")
#     return models

# models = load_models()

# # ==================== PREPROCESS & GRAD-CAM ====================
# def preprocess(img):
#     img_resized = img.resize((224, 224))
#     arr = image.img_to_array(img_resized) / 255.0
#     return np.expand_dims(arr, axis=0), np.array(img_resized)

# def get_gradcam_plus_plus(model, img_array, layer_name=None):
#     try:
#         # Auto-find last conv layer
#         for layer in reversed(model.layers):
#             if len(layer.output_shape) == 4 and ('conv' in layer.name or 'block' in layer.name):
#                 last_conv_layer = layer
#                 break
#         else:
#             return np.zeros((224, 224, 3), np.uint8)

#         grad_model = tf.keras.models.Model(
#             [model.input], [last_conv_layer.output, model.output]
#         )

#         with tf.GradientTape() as tape:
#             conv_outputs, predictions = grad_model(img_array)
#             class_idx = tf.argmax(predictions[0])
#             loss = predictions[:, class_idx]

#         grads = tape.gradient(loss, conv_outputs)
#         conv_outputs = conv_outputs[0]
#         grads = grads[0]

#         # Grad-CAM++ weighting
#         alpha_num = grads**2
#         alpha_denom = 2.0 * grads**2 + tf.reduce_sum(conv_outputs * (grads**3), axis=[0, 1])
#         alpha_denom = tf.where(alpha_denom != 0.0, alpha_denom, tf.ones_like(alpha_denom))
#         alphas = alpha_num / (alpha_denom + 1e-8)

#         weights = tf.maximum(grads, 0) * alphas
#         weights = tf.reduce_sum(weights, axis=[0, 1])

#         cam = tf.reduce_sum(weights[tf.newaxis, tf.newaxis, :] * conv_outputs, axis=-1)
#         cam = tf.maximum(cam, 0)
#         cam = cam / (tf.reduce_max(cam) + 1e-8)
#         cam = cv.resize(cam.numpy(), (224, 224))
#         cam = np.uint8(255 * cam)
#         heatmap = cv.applyColorMap(cam, cv.COLORMAP_JET)
#         return heatmap
#     except:
#         return np.zeros((224, 224, 3), np.uint8)

# # ==================== MAIN UI ====================
# st.markdown("<h1 class='header-title'>DERMALYTICS</h1>", unsafe_allow_html=True)
# st.markdown("<p class='subtitle'>Advanced AI-Powered Skin Lesion Classification with Explainable Grad-CAM++ Visualization</p>", unsafe_allow_html=True)

# st.markdown("---")

# if not models:
#     st.error("No trained models were loaded. Please ensure `.h5` files are in the `models/` directory.")
#     st.stop()

# col1, col2 = st.columns([2, 1])
# with col1:
#     selected_model_name = st.selectbox("Select AI Model", options=list(models.keys()), index=0)
# with col2:
#     st.markdown("<br>", unsafe_allow_html=True)
#     st.info(f"**Loaded:** {selected_model_name}")

# model = models[selected_model_name]

# st.markdown("### Upload Dermoscopic Image")
# uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "bmp"], label_visibility="collapsed")

# if uploaded_file:
#     image_pil = Image.open(uploaded_file).convert("RGB")
    
#     col_a, col_b = st.columns([1, 1])
#     with col_a:
#         st.image(image_pil, caption="Uploaded Lesion Image", width=350)
    
#     with col_b:
#         st.markdown("#### Processing...")
#         input_array, display_img = preprocess(image_pil)
        
#         with st.spinner("Running inference & generating explanation..."):
#             prediction = model.predict(input_array, verbose=0)[0]
#             pred_idx = np.argmax(prediction)
#             confidence = float(prediction[pred_idx])
#             heatmap = get_gradcam_plus_plus(model, input_array)
#             overlay = cv.addWeighted(display_img, 0.65, heatmap, 0.35, 0)

#         # Diagnosis Card
#         st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
#         st.markdown(f"### Diagnosis: **{CLASS_NAMES[SHORT_NAMES[pred_idx]]}**")
#         risk_level = "HIGH RISK" if SHORT_NAMES[pred_idx] in ["mel", "bcc", "akiec"] else "LOW RISK"
#         st.markdown(f"**Predicted Class:** `{SHORT_NAMES[pred_idx].upper()}` • **Risk Level:** <span style='color:#e74c3c;font-weight:bold'>{risk_level}</span>", unsafe_allow_html=True)
#         st.metric("Confidence", f"{confidence:.1%}")
#         st.markdown("</div>", unsafe_allow_html=True)

#         # Charts
#         st.markdown("#### Probability Distribution")
#         chart_data = {CLASS_NAMES[k]: float(v) for k, v in zip(SHORT_NAMES, prediction)}
#         st.bar_chart(chart_data, height=300, use_container_width=True)

#     st.markdown("---")
#     st.markdown("### Explainable AI: Grad-CAM++ Attention Map")

#     col1, col2 = st.columns(2)
#     with col1:
#         st.image(heatmap, caption="Heatmap (Where the AI is looking)", channels="BGR", use_column_width=True)
#     with col2:
#         st.image(overlay, caption="Overlay on Original Image", use_column_width=True)

#     st.info("Red areas indicate regions the model focused on most when making its prediction. This increases transparency and trust in AI diagnosis.")

# st.markdown("---")
# st.markdown("""
# <div class='footer'>
#     <strong>DERMALYTICS</strong> • AI Research Project • Trained on HAM10000 Dataset<br>
#  <em>Note: This tool is for research and educational purposes only. Not for clinical diagnosis.</em>
# </div>
# """, unsafe_allow_html=True)

import streamlit as st
import streamlit_option_menu 
st.title("Welcome")