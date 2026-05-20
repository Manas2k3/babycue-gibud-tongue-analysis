import json
import torch
import timm
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class names from JSON
with open("models/class_names.json", "r") as f:
    COLOR_CLASSES = json.load(f)   # e.g. ['deep_red','healthy','indigo_violet','purple','white']

# ------------------------------
# Advanced colour analysis parameters (from notebook)
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
    """Compute HSV statistics from RGB numpy array. Returns dict with native floats."""
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(float)
    h, s, v = cv2.split(hsv)
    return {
        'mh':   float(np.median(h)),
        'ms':   float(np.median(s)),
        'mv':   float(np.median(v)),
        'p25v': float(np.percentile(v, 25)),
        'p75v': float(np.percentile(v, 75)),
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
    ratio = float(sum(blobs) / total_pixels) if total_pixels > 0 else 0.0
    return bool(ratio > COAT_THRESHOLD), ratio

def refine(cls, st):
    """Refine model prediction using HSV rules (from notebook)."""
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
    """Load trained EfficientNet‑V2 colour model."""
    model = timm.create_model('tf_efficientnetv2_b2.in1k', pretrained=False, num_classes=len(COLOR_CLASSES))
    ckpt = torch.load("models/efficientnetv2_tongue_color.pth", map_location=DEVICE)
    if "model_state" in ckpt:
        ckpt = ckpt["model_state"]
    model.load_state_dict(ckpt)
    model.to(DEVICE).eval()
    return model

def predict_color(model, pil_img):
    """
    Run full advanced colour pipeline.
    Returns dict with: final_colour, base_colour, confidence, coated, coating_ratio.
    """
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    # Model inference
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()
    idx = int(np.argmax(probs))
    raw_class = COLOR_CLASSES[idx]
    confidence = probs[idx]

    # HSV stats and coating detection
    img_np = np.array(pil_img.convert('RGB'))
    st = hsv_stats(img_np)
    coated, ratio = coating_ratio(img_np)

    # Refine and apply coating override
    base_colour = refine(raw_class, st)
    final_colour = coating_override(base_colour, coated, ratio, st)

    return {
        'final_colour': final_colour,
        'base_colour': base_colour,
        'confidence': round(confidence * 100, 2),
        'coated': coated,
        'coating_ratio': ratio,
    }

# Legacy function kept for compatibility with existing code that expects detect_coating
def detect_coating(pil_img, white_v_min=200, white_s_max=40, coat_threshold=0.08, min_blob_area=400):
    """Legacy function: returns {"label": "white" if coated else "none"}."""
    img_np = np.array(pil_img.convert("RGB"))
    coated, _ = coating_ratio(img_np)
    return {"label": "white" if coated else "none"}
