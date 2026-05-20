import os
import logging
from flask import Flask, request, jsonify
from utils.image_utils import load_image_from_bytes, encode_image
from utils.color_utils import load_color_model, predict_color, detect_coating
from utils.texture_utils import load_texture_model, predict_texture
from utils.shape_utils import load_shape_cnn, load_shape_ml, predict_shape
from utils.crack_utils import load_crack_model, predict_crack

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB
logging.basicConfig(level=logging.INFO)

# Global model cache
models = {}

def load_all_models():
    logging.info("Loading colour model...")
    models["color"] = load_color_model()
    logging.info("Loading texture model...")
    models["texture"] = load_texture_model()
    logging.info("Loading shape CNN and ML ensemble...")
    models["shape_cnn"], models["shape_classes"] = load_shape_cnn()
    models["shape_ml"] = load_shape_ml()
    logging.info("Loading crack model...")
    models["crack"] = load_crack_model()
    logging.info("All models loaded.")

def compute_confidence(color_label, texture_label, shape_label, crack_label, coating_label):
    """Heuristic confidence score – replace with LR if available."""
    color_weights = {"pink":1.0, "white":0.5, "red":0.35, "purple":0.25, "indigo_violet":0.2}
    texture_weights = {"normal":1.0, "tender":0.5, "tough":0.35}
    shape_weights = {"normal":1.0, "oval":0.7, "rectangle":0.6}
    crack_weights = {"non_cracked":1.0, "cracked":0.3}
    coating_weights = {"none":1.0, "white":0.4}
    score = (color_weights.get(color_label, 0.5) * 0.25 +
             texture_weights.get(texture_label, 0.5) * 0.20 +
             shape_weights.get(shape_label, 0.7) * 0.20 +
             crack_weights.get(crack_label, 0.5) * 0.20 +
             coating_weights.get(coating_label, 0.5) * 0.15)
    if score >= 0.8: band = "Almost Perfect"
    elif score >= 0.6: band = "Substantial"
    elif score >= 0.4: band = "Moderate"
    elif score >= 0.2: band = "Fair"
    else: band = "Poor"
    return {"score": round(score,4), "band": band}

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": str(DEVICE) if "color" in models else "unknown"}), 200

@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No image file provided"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "Empty filename"}), 400

    try:
        pil_img = load_image_from_bytes(file.read())
    except Exception as e:
        return jsonify({"status": "error", "message": f"Invalid image: {str(e)}"}), 400

    # Run predictions with error handling
    try:
        color_result = predict_color(models["color"], pil_img)
    except Exception as e:
        logging.error(f"Color model error: {e}")
        color_result = {"label": "error"}

    try:
        texture_result = predict_texture(models["texture"], pil_img)
    except Exception as e:
        logging.error(f"Texture model error: {e}")
        texture_result = {"label": "error"}

    try:
        shape_result = predict_shape(models["shape_cnn"], models["shape_classes"], models["shape_ml"], pil_img)
    except Exception as e:
        logging.error(f"Shape model error: {e}")
        shape_result = {"label": "error"}

    try:
        crack_result = predict_crack(models["crack"], pil_img)
    except Exception as e:
        logging.error(f"Crack model error: {e}")
        crack_result = {"label": "error"}

    try:
        coating_result = detect_coating(pil_img)
    except Exception as e:
        logging.error(f"Coating detection error: {e}")
        coating_result = {"label": "error"}

    # Confidence score
    confidence = compute_confidence(
        color_result["label"], texture_result["label"],
        shape_result["label"], crack_result["label"], coating_result["label"]
    )

    # Encode image
    img_base64 = encode_image(pil_img)

    response = {
        "status": "success",
        "uploaded_image": img_base64,
        "color": color_result,
        "texture": texture_result,
        "shape": shape_result,
        "cracks": crack_result,
        "coating": coating_result,
        "confidence_score": confidence
    }
    return jsonify(response), 200

if __name__ == "__main__":
    load_all_models()
    app.run(host="0.0.0.0", port=5000, debug=False)
