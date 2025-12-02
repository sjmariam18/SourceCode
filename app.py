# # app.py
# from flask import Flask, render_template, request, flash, redirect, url_for
# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from PIL import Image
# import cv2
# import base64
# import traceback

# # Force CPU (remove if you have GPU and want to use it)
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# app = Flask(__name__)
# app.secret_key = "DERMALYTICS"
# app.config['UPLOAD_FOLDER'] = 'static/uploads'
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# # ==================== CONFIG ====================
# CLASS_NAMES = {
#     0: "akiec", 1: "bcc", 2: "bkl", 3: "df",
#     4: "mel", 5: "nv", 6: "vasc"
# }

# MODEL_PATHS = {
#     "Model 1 (Recommended)": r"C:\Users\lenovo\Documents\CAPSTONE\CAPSTONE C\SourceCode\models\best_model.h5",
#     "Model 2": r"C:\Users\lenovo\Documents\CAPSTONE\CAPSTONE C\SourceCode\models\Xception_HAM10000.h5"
# }

# # Load models
# models = {}
# for name, path in MODEL_PATHS.items():
#     if os.path.exists(path):
#         print(f"Loading {name} from {path}...")
#         try:
#             model = load_model(path, compile=False)
#             models[name] = model
#             print(f"Model {name} loaded successfully!")
#             print("Model layers (last 20):")
#             for i, layer in enumerate(model.layers[-20:]):
#                 print(f"  {len(model.layers)-20+i:3d}: {layer.name:35s} {str(layer.output.shape):20s} {type(layer).__name__}")
#         except Exception as e:
#             print(f"Failed to load {name}: {e}")
#     else:
#         print(f"Warning: Model file not found: {path}")

# if not models:
#     raise FileNotFoundError("No valid models were loaded! Check MODEL_PATHS.")

# current_model = list(models.values())[0]

# # ==================== HELPER: Find last Conv2D layer ====================
# def find_last_conv_layer(model):
#     """Safely find the last Conv2D layer (most reliable for Grad-CAM)"""
#     for layer in reversed(model.layers):
#         if isinstance(layer, tf.keras.layers.Conv2D):
#             return layer.name
    
#     # Fallback: any 4D layer with "conv" in name
#     for layer in reversed(model.layers):
#         if len(layer.output.shape) == 4 and 'conv' in layer.name.lower():
#             return layer.name
    
#     # Last resort
#     for layer in reversed(model.layers):
#         if len(layer.output.shape) == 4 and layer.output.shape[-1] >= 32:
#             return layer.name
    
#     raise ValueError("No suitable convolutional layer found for Grad-CAM!")

# # ==================== PREPROCESS ====================
# def preprocess_image(img_path):
#     img = Image.open(img_path).convert("RGB")
#     img = img.resize((224, 224))
#     arr = image.img_to_array(img) / 255.0
#     arr = np.expand_dims(arr, axis=0)
#     return arr, np.array(img)  # returns (1,224,224,3) and (224,224,3)

# # ==================== GRAD-CAM++ ====================
# # ==================== GRAD-CAM++ (加强版：明确告诉你用了哪一层) ====================
# def get_gradcam_plus_plus(model, img_array, target_layer_name=None):
#     try:
#         if target_layer_name is None:
#             target_layer_name = find_last_conv_layer(model)

#         # 重写这部分：明确打印所选层的信息
#         target_layer = model.get_layer(target_layer_name)
#         layer_output_shape = target_layer.output.shape  # 正确方式获取 shape
#         print("\n" + "="*60)
#         print("GRAD-CAM++ ACTIVATION LAYER SELECTED")
#         print("="*60)
#         print(f"Layer Name       : {target_layer_name}")
#         print(f"Layer Type       : {type(target_layer).__name__}")
#         print(f"Output Shape     : {layer_output_shape}")
#         print(f"Channels         : {layer_output_shape[-1]}")
#         print("="*60 + "\n")

#         grad_model = tf.keras.models.Model(
#             inputs=model.inputs,
#             outputs=[target_layer.output, model.output]
#         )

#         with tf.GradientTape() as tape:
#             conv_outputs, predictions = grad_model(img_array, training=False)
#             class_idx = tf.argmax(predictions[0])
#             loss = predictions[:, class_idx]

#         grads = tape.gradient(loss, conv_outputs)
#         if grads is None:
#             grads = tf.zeros_like(conv_outputs)

#         grads = tf.nn.relu(grads)
#         weights = tf.reduce_mean(grads, axis=(1, 2))
#         weights = tf.nn.relu(weights)

#         cam = tf.reduce_sum(weights[:, None, None, :] * conv_outputs[0], axis=-1)
#         cam = tf.nn.relu(cam)
#         cam = cam.numpy()

#         if cam.max() > 0:
#             cam = cam / cam.max()
#         cam = np.uint8(255 * cam)
#         cam = cv2.resize(cam, (224, 224))
#         cam_colored = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

#         return cam_colored, int(class_idx), predictions.numpy()[0], target_layer_name

#     except Exception as e:
#         print("Grad-CAM++ failed:", str(e))
#         traceback.print_exc()
#         dummy = np.zeros((224, 224), dtype=np.uint8)
#         dummy_colored = cv2.applyColorMap(dummy, cv2.COLORMAP_JET)
#         return dummy_colored, 0, np.zeros(7), "ERROR"

# # ==================== Overlay ====================
# def overlay_heatmap(original_bgr, heatmap, alpha=0.5):
#     return cv2.addWeighted(original_bgr, 0.6, heatmap, alpha, 0)

# # ==================== ROUTES ====================
# @app.route("/", methods=["GET", "POST"])
# def index():
#     result = None
#     selected_model = request.form.get("model_select", list(models.keys())[0])

#     if request.method == "POST":
#         if "file" not in request.files:
#             flash("No file part")
#             return redirect(request.url)

#         file = request.files["file"]
#         if file.filename == "":
#             flash("No file selected")
#             return redirect(request.url)

#         allowed_ext = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}
#         if not (file and file.filename.lower().split('.')[-1] in allowed_ext):
#             flash("Invalid file type")
#             return redirect(request.url)

#         # Save uploaded image
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], "latest_upload.jpg")
#         file.save(filepath)

#         # Select model
#         model = models.get(selected_model, current_model)

#         try:
#             # Preprocess
#             img_array, display_img_rgb = preprocess_image(filepath)

#             # Prediction
#             pred_probs = model.predict(img_array, verbose=0)[0]
#             class_id = int(np.argmax(pred_probs))
#             confidence = float(pred_probs[class_id])

#             # Grad-CAM++
#             heatmap_colored, cam_class_id, probs, layer_name = get_gradcam_plus_plus(model, img_array)

#             # Overlay
#             original_bgr = cv2.cvtColor(display_img_rgb, cv2.COLOR_RGB2BGR)
#             overlay_bgr = overlay_heatmap(original_bgr, heatmap_colored, alpha=0.5)
#             overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

#             # Convert to base64
#             def img_to_b64(img_rgb):
#                 _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
#                 return base64.b64encode(buffer).decode('utf-8')

#             result = {
#                 "original": f"/static/uploads/latest_upload.jpg?t={os.path.getmtime(filepath)}",
#                 "overlay": "data:image/jpeg;base64," + img_to_b64(overlay_rgb),
#                 "heatmap": "data:image/jpeg;base64," + img_to_b64(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)),
#                 "prediction": CLASS_NAMES.get(class_id, "Unknown"),
#                 "confidence": round(confidence * 100, 2),
#                 "probabilities": [(CLASS_NAMES.get(i, "Unknown"), f"{p*100:.2f}%") for i, p in enumerate(probs)],
#                 "layer": layer_name,
#                 "model_used": selected_model
#             }

#         except Exception as e:
#             print("Processing error:", str(e))
#             traceback.print_exc()
#             flash("Error processing image. Check server console.")
#             result = None

#     return render_template(
#         "index.html",
#         result=result,
#         models=models.keys(),
#         selected_model=selected_model
#     )

# if __name__ == "__main__":
#     print("Skin Lesion Classifier Flask App Running")
#     print("Visit: http://127.0.0.1:5000")
#     app.run(debug=False, host="0.0.0.0", port=5000)

    
# app.py
from flask import Flask, render_template, request, flash, redirect, url_for
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
import base64
import traceback

# Force CPU only (remove if you have GPU)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app = Flask(__name__)
app.secret_key = "DERMALYTICS"
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
    "Model 7 (EfficientNetB3)"   : r"C:\Users\Users\lenovo\Documents\CAPSTONE\CAPSTONE C\SourceCode\models\EfficientNetB3_HAM10000.h5",
    "Model 8 (ResNet101)"        : r"C:\Users\lenovo\Documents\CAPSTONE\CAPSTONE C\SourceCode\models\ResNet101_HAM10000.h5",
    "Model 9 (ResNet50)"         : r"C:\Users\lenovo\Documents\CAPSTONE\CAPSTONE C\SourceCode\models\ResNet50_HAM10000.h5"
}

# Load all models
models = {}
for name, path in MODEL_PATHS.items():
    if os.path.exists(path):
        print(f"Loading {name} from {path}...")
        try:
            model = load_model(path, compile=False)
            models[name] = model
            print(f"{name} loaded successfully!")
        except Exception as e:
            print(f"Failed to load {name}: {e}")
    else:
        print(f"Model not found: {path}")

if not models:
    raise FileNotFoundError("No models loaded! Check MODEL_PATHS.")

# Default model
current_model_name = list(models.keys())[0]
current_model = models[current_model_name]

# ==================== HELPER: Find last Conv2D ====================
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            return layer.name
    raise ValueError("No convolutional layer found for Grad-CAM!")

# ==================== PREPROCESS IMAGE ====================
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img = img.resize((224, 224))
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr, np.array(img)  # (1,224,224,3), (224,224,3)

# ==================== TRUE GRAD-CAM++ (Fixed & Robust) ====================
def get_gradcam_plus_plus(model, img_array, target_layer_name=None):
    try:
        if target_layer_name is None:
            target_layer_name = find_last_conv_layer(model)

        target_layer = model.get_layer(target_layer_name)
        print("\n" + "="*60)
        print("GRAD-CAM++ ACTIVATION LAYER")
        print("="*60)
        print(f"Layer Name   : {target_layer_name}")
        print(f"Layer Type   : {type(target_layer).__name__}")
        print(f"Output Shape : {target_layer.output.shape}")
        print(f"Channels     : {target_layer.output.shape[-1]}")
        print("="*60 + "\n")

        # Handle model.output being a list (common in old .h5 files)
        model_output = model.output
        if isinstance(model_output, list):
            model_output = model_output[0]

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model_output]
        )

        img_array = tf.cast(img_array, tf.float32)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            # Handle predictions list
            if isinstance(predictions, (list, tuple)):
                predictions = predictions[0]
            predictions = tf.squeeze(predictions)

            class_idx = tf.argmax(predictions)
            loss = predictions[class_idx]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            grads = tf.zeros_like(conv_outputs)

        # === True Grad-CAM++ (second-order weights) ===
        grads = tf.cast(grads, tf.float32)
        conv_outputs = tf.cast(conv_outputs, tf.float32)

        # Compute alphas (Grad-CAM++ specific)
        with tf.GradientTape() as tape2:
            tape2.watch(conv_outputs)
            recreated_loss = tf.gather(predictions, class_idx)
        second_grads = tape2.gradient(recreated_loss, conv_outputs)
        if second_grads is None:
            second_grads = tf.zeros_like(grads)

        alpha_num = tf.pow(grads, 2)
        alpha_denom = 2.0 * tf.pow(grads, 2) + tf.reduce_sum(
            tf.pow(second_grads, 3) * conv_outputs, axis=[1, 2], keepdims=True
        )
        alpha_denom = tf.where(alpha_denom == 0.0, tf.ones_like(alpha_denom) * 1e-10, alpha_denom)
        alphas = alpha_num / alpha_denom

        weights = tf.nn.relu(grads) * alphas
        weights = tf.reduce_sum(weights, axis=(1, 2))

        cam = tf.reduce_sum(weights[:, None, None, :] * conv_outputs[0], axis=-1)
        cam = tf.nn.relu(cam)
        cam = cam.numpy()

        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = np.uint8(255 * cam)
        cam = cv2.resize(cam, (224, 224))
        cam_colored = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

        # Safe probabilities
        pred_np = predictions.numpy() if hasattr(predictions, 'numpy') else np.array(predictions)
        if pred_np.ndim > 1:
            pred_np = pred_np.flatten()

        return cam_colored, int(class_idx), pred_np, target_layer_name

    except Exception as e:
        print("Grad-CAM++ Error:", str(e))
        traceback.print_exc()
        dummy = np.zeros((224, 224), dtype=np.uint8)
        dummy_colored = cv2.applyColorMap(dummy, cv2.COLORMAP_JET)
        return dummy_colored, 0, np.zeros(7), "ERROR"

# ==================== OVERLAY HEATMAP ====================
def overlay_heatmap(original_bgr, heatmap, alpha=0.5):
    return cv2.addWeighted(original_bgr, 0.6, heatmap, alpha, 0)

# ==================== IMAGE TO BASE64 ====================
def img_to_b64(img_rgb):
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')

# ==================== ROUTES ====================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    selected_model_name = request.form.get("model_select", current_model_name)

    if request.method == "POST":
        if "file" not in request.files:
            flash("No file uploaded")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("No file selected")
            return redirect(request.url)

        # File type check
        allowed = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}
        if not file.filename.lower().split('.')[-1] in allowed:
            flash("Invalid file type")
            return redirect(request.url)

        # Save image
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], "latest_upload.jpg")
        file.save(filepath)

        # Select model
        model = models.get(selected_model_name, current_model)

        try:
            img_array, display_img_rgb = preprocess_image(filepath)

            # === Safe Prediction ===
            raw_pred = model.predict(img_array, verbose=0)
            if isinstance(raw_pred, list):
                raw_pred = raw_pred[0]
            pred_probs = raw_pred.flatten()
            class_id = int(np.argmax(pred_probs))
            confidence = float(pred_probs[class_id])

            # === Grad-CAM++ ===
            heatmap_colored, cam_class_id, cam_probs, layer_name = get_gradcam_plus_plus(model, img_array)

            # Overlay
            original_bgr = cv2.cvtColor(display_img_rgb, cv2.COLOR_RGB2BGR)
            overlay_bgr = overlay_heatmap(original_bgr, heatmap_colored, alpha=0.5)
            overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

            # Results dict
            result = {
                "original": f"/static/uploads/latest_upload.jpg?t={os.path.getmtime(filepath)}",
                "overlay": "data:image/jpeg;base64," + img_to_b64(overlay_rgb),
                "heatmap": "data:image/jpeg;base64," + img_to_b64(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)),
                "prediction": CLASS_NAMES.get(class_id, "Unknown"),
                "confidence": confidence,  # FIXED: keep raw probability (0–1)
                "probabilities": [(CLASS_NAMES.get(i, f"Class {i}"), f"{p*100:.2f}%") for i, p in enumerate(pred_probs)],
                "layer": layer_name,
                "model_used": selected_model_name
            }


        except Exception as e:
            print("Processing failed:", str(e))
            traceback.print_exc()
            flash("Error processing image. See console.")
            result = None

    return render_template(
        "index.html",
        result=result,
        models=models.keys(),
        selected_model=selected_model_name
    )

# ==================== RUN ====================
if __name__ == "__main__":
    print("Skin Lesion Classifier (HAM10000/ISIC) with Grad-CAM++")
    print("Available models:", list(models.keys()))
    print("Visit: http://127.0.0.1:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)