import os
import io
import logging
import numpy as np
import cv2
import torch
import timm
from flask import Flask, request, jsonify
from PIL import Image
import base64
from torchvision import transforms

# ------------------------------
# Flask setup
# ------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB
logging.basicConfig(level=logging.INFO)

# ------------------------------
# Global model cache
# ------------------------------
models = {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Advanced tongue colour analysis
# ------------------------------
WHITE_V_MIN      = 200
WHITE_S_MAX      = 40
COAT_THRESHOLD   = 0.08
MIN_BLOB_AREA    = 400

COAT_LIMITS = {
    "deep red": 0.25, "red": 0.20, "dark red purple": 0.22,
    "pink": 0.15, "pale red": 0.10, "pale pink": 0.09,
    "pale": 0.08, "white": 0.08,
    "purple": 0.18, "pale purple": 0.15,
    "indigo violet": 0.18, "pale indigo violet": 0.15,
    "blue purple": 0.18,
}

def hsv_stats(img_np):
    """Compute HSV statistics from RGB numpy array."""
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(float)
    h, s, v = cv2.split(hsv)
    return {
        'mh':   float(np.median(h)),
        'ms':   float(np.median(s)),
        'mv':   float(np.median(v)),
        'p75s': float(np.percentile(s, 75)),
    }

def coating_ratio(img_np):
    """Detect white coating and return (coated_bool, ratio)."""
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (0, 0, WHITE_V_MIN), (180, WHITE_S_MAX, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    _, labels = cv2.connectedComponents(mask)
    blobs = [np.sum(labels == i) for i in range(1, labels.max() + 1)
             if np.sum(labels == i) > MIN_BLOB_AREA]
    total_pixels = img_np.shape[0] * img_np.shape[1]
    ratio = float(sum(blobs) / total_pixels)
    return bool(ratio > COAT_THRESHOLD), ratio

def refine(cls, st):
    """Refine model prediction using HSV rules."""
    mh, ms, mv = st['mh'], st['ms'], st['mv']
    red_hue    = (mh <= 15) or (mh >= 168)
    purple_hue = (110 <= mh <= 165)

    def purple_family(h, v):
        if h < 131:          base = "purple"
        elif h < 159:        base = "indigo violet"
        else:                base = "blue purple"
        if v >= 185 and base == "indigo violet": return "pale indigo violet"
        if v >= 185 and base == "purple":        return "pale purple"
        return base

    # Purple/violet hue overrides model
    if purple_hue and ms >= 25:
        return purple_family(mh, mv)

    # Very bright surface
    if mv >= 200:
        if red_hue and ms < 90:  return "pale pink"
        if ms < 90:              return "pale"
        return "pale red"
    if mv >= 160 and ms < 80:
        return "pale pink"

    # Per-class logic
    if cls == "deep_red":
        if ms < 70 or (mv >= 150 and ms < 100): return "pale"
        if red_hue and ms >= 120 and mv < 160:  return "deep red"
        if red_hue and ms >= 100 and mv >= 160: return "red"
        if ms >= 80 and mv < 120:               return "deep red"
        return "deep red"

    if cls == "healthy":
        if ms < 60:                              return "pale pink"
        if red_hue and ms >= 160 and mv < 150:  return "deep red"
        if red_hue and ms >= 120 and mv < 160:  return "deep red"
        if ms >= 100 and mv >= 160:             return "red"
        if ms >= 100 and red_hue:               return "deep red"
        return "pink"

    if cls == "white":
        if purple_hue and ms >= 20:             return purple_family(mh, mv)
        if ms > 90:                             return "pale red"
        if ms > 55:                             return "pale pink"
        return "white"

    if cls == "indigo_violet":
        if red_hue and ms >= 120 and mv < 160:  return "deep red"
        if red_hue and ms >= 80:                return "red"
        if not purple_hue:                      return "red"
        return purple_family(mh, mv)

    if cls == "purple":
        if red_hue and ms >= 120 and mv < 160:  return "deep red"
        if red_hue and ms >= 80:                return "red"
        if mh < 128:                            return "dark red purple"
        return purple_family(mh, mv)

    return cls

def coating_override(colour, coated, ratio, st):
    """Apply coating detection override."""
    if not coated:
        return colour
    red_hue = (st['mh'] <= 15) or (st['mh'] >= 168)
    p75s = st['p75s']
    if red_hue and colour in ("pale pink", "pale", "pale red", "pink"):
        sat_spread = p75s - st['ms']
        if sat_spread >= 25 and ratio >= 0.18:
            return "white coated"
        elif ratio >= 0.40:
            return "white coated"
        else:
            return colour
    return "white coated" if ratio >= COAT_LIMITS.get(colour, COAT_THRESHOLD) else colour

def load_color_model():
    """Load EfficientNetV2-B2 colour model."""
    MODEL_PATH = "efficientnetv2_tongue_color_50epochs.pth"
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    class_names = checkpoint['class_names']
    img_size = checkpoint.get('img_size', 224)
    backbone = checkpoint.get('backbone', 'tf_efficientnetv2_b2.in1k')
    model = timm.create_model(backbone, pretrained=False, num_classes=len(class_names))
    model.load_state_dict(checkpoint['model_state'])
    model.to(DEVICE)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return {
        'model': model,
        'class_names': class_names,
        'transform': transform,
        'img_size': img_size
    }

def predict_advanced_color(color_model, pil_img):
    """
    Run full colour pipeline.
    Returns dict with only: final_colour, base_colour, confidence, coated, coating_ratio.
    """
    # Model inference
    tensor = color_model['transform'](pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = color_model['model'](tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()
    idx = probs.index(max(probs))
    raw_class = color_model['class_names'][idx]
    confidence = probs[idx]

    # HSV and coating
    img_np = np.array(pil_img.convert('RGB'))
    st = hsv_stats(img_np)
    coated, ratio = coating_ratio(img_np)

    # Refine and override
    base_colour = refine(raw_class, st)
    final_colour = coating_override(base_colour, coated, ratio, st)

    return {
        'final_colour': final_colour,
        'base_colour': base_colour,
        'confidence': round(confidence * 100, 2),
        'coated': coated,
        'coating_ratio': ratio,
    }

# ------------------------------
# Dummy placeholders for other models (replace with actual imports)
# ------------------------------
def load_texture_model():
    # Placeholder – replace with your actual texture model loading
    return None

def load_shape_cnn():
    return None, None

def load_shape_ml():
    return None

def load_crack_model():
    return None

def predict_texture(model, pil_img):
    return {"label": "normal"}  # Placeholder

def predict_shape(cnn_model, classes, ml_model, pil_img):
    return {"label": "normal"}  # Placeholder

def predict_crack(model, pil_img):
    return {"label": "non_cracked"}  # Placeholder

def detect_coating(pil_img):
    # Already handled in advanced colour pipeline; return placeholder
    return {"label": "none"}

def encode_image(pil_img):
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

# ------------------------------
# Load all models
# ------------------------------
def load_all_models():
    logging.info("Loading colour model (advanced pipeline)...")
    models["color"] = load_color_model()
    logging.info("Loading texture model...")
    models["texture"] = load_texture_model()
    logging.info("Loading shape CNN and ML ensemble...")
    models["shape_cnn"], models["shape_classes"] = load_shape_cnn()
    models["shape_ml"] = load_shape_ml()
    logging.info("Loading crack model...")
    models["crack"] = load_crack_model()
    logging.info("All models loaded.")

# ------------------------------
# Confidence score (unchanged)
# ------------------------------
def compute_confidence(color_label, texture_label, shape_label, crack_label, coating_label):
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

# ------------------------------
# Routes
# ------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": str(DEVICE)}), 200

@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No image file provided"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "Empty filename"}), 400

    try:
        pil_img = Image.open(io.BytesIO(file.read())).convert('RGB')
    except Exception as e:
        return jsonify({"status": "error", "message": f"Invalid image: {str(e)}"}), 400

    # Colour analysis (advanced)
    try:
        color_result = predict_advanced_color(models["color"], pil_img)
    except Exception as e:
        logging.error(f"Colour model error: {e}")
        color_result = {"final_colour": "error", "base_colour": "error", "confidence": 0, "coated": False, "coating_ratio": 0.0}

    # Other models (with fallbacks)
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

    # Overall confidence (using refined final_colour as colour label)
    overall_confidence = compute_confidence(
        color_result["final_colour"], 
        texture_result["label"], 
        shape_result["label"], 
        crack_result["label"], 
        coating_result["label"]
    )

    # Encode image for response
    img_base64 = encode_image(pil_img)

    response = {
        "status": "success",
        "uploaded_image": img_base64,
        "color": color_result,        # contains final_colour, base_colour, confidence, coated, coating_ratio
        "texture": texture_result,
        "shape": shape_result,
        "cracks": crack_result,
        "coating": coating_result,
        "confidence_score": overall_confidence
    }
    return jsonify(response), 200

if __name__ == "__main__":
    load_all_models()
    app.run(host="0.0.0.0", port=5000, debug=False)
