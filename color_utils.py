import os
import numpy as np
import cv2
import torch
import timm
from torchvision import transforms

# ------------------------------
# Constants for coating detection
# ------------------------------
WHITE_V_MIN    = 200      # minimum Value for white coating
WHITE_S_MAX    = 40       # maximum Saturation for white coating
COAT_THRESHOLD = 0.08     # 8% coating area triggers detection
MIN_BLOB_AREA  = 400      # ignore tiny specks

# Per‑colour thresholds for coating override (from your original)
COAT_LIMITS = {
    "deep red": 0.25, "red": 0.20, "dark red purple": 0.22,
    "pink": 0.15, "pale red": 0.10, "pale pink": 0.09,
    "pale": 0.08, "white": 0.08,
    "purple": 0.18, "pale purple": 0.15,
    "indigo violet": 0.18, "pale indigo violet": 0.15,
    "blue purple": 0.18,
}

# ------------------------------
# Helper: HSV statistics
# ------------------------------
def hsv_stats(img_np):
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(float)
    h, s, v = cv2.split(hsv)
    return {
        'mh':   float(np.median(h)),
        'ms':   float(np.median(s)),
        'mv':   float(np.median(v)),
        'p75s': float(np.percentile(s, 75)),
    }

# ------------------------------
# Coating detection (white/yellow)
# ------------------------------
def coating_ratio(img_np):
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    # White coating mask
    mask = cv2.inRange(hsv, (0, 0, WHITE_V_MIN), (180, WHITE_S_MAX, 255))
    # Yellowish coating (hue 20-30, low sat, high value)
    mask_yellow = cv2.inRange(hsv, (20, 30, 180), (30, 80, 255))
    mask = cv2.bitwise_or(mask, mask_yellow)
    # Morphological cleanup
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # Remove tiny connected components
    _, labels = cv2.connectedComponents(mask)
    blobs = [
        np.sum(labels == i)
        for i in range(1, labels.max() + 1)
        if np.sum(labels == i) > MIN_BLOB_AREA
    ]
    total_pixels = img_np.shape[0] * img_np.shape[1]
    ratio = float(sum(blobs) / total_pixels) if total_pixels > 0 else 0.0
    coated = ratio > COAT_THRESHOLD
    return coated, ratio

# ------------------------------
# Refine raw model output to base colour
# ------------------------------
def refine(raw_class, stats):
    mh, ms, mv = stats['mh'], stats['ms'], stats['mv']
    red_hue    = (mh <= 15) or (mh >= 168)
    purple_hue = (110 <= mh <= 165)

    def purple_family(h, v):
        if h < 131:
            base = "purple"
        elif h < 159:
            base = "indigo violet"
        else:
            base = "blue purple"
        if v >= 185 and base == "indigo violet":
            return "pale indigo violet"
        if v >= 185 and base == "purple":
            return "pale purple"
        return base

    if purple_hue and ms >= 25:
        return purple_family(mh, mv)

    if mv >= 200:
        if red_hue and ms < 90:
            return "pale pink"
        if ms < 90:
            return "pale"
        return "pale red"
    if mv >= 160 and ms < 80:
        return "pale pink"

    if raw_class == "deep_red":
        if ms < 70 or (mv >= 150 and ms < 100):
            return "pale"
        if red_hue and ms >= 120 and mv < 160:
            return "deep red"
        if red_hue and ms >= 100 and mv >= 160:
            return "red"
        if ms >= 80 and mv < 120:
            return "deep red"
        return "deep red"

    if raw_class == "healthy":
        if ms < 60:
            return "pale pink"
        if red_hue and ms >= 160 and mv < 150:
            return "deep red"
        if red_hue and ms >= 120 and mv < 160:
            return "deep red"
        if ms >= 100 and mv >= 160:
            return "red"
        if ms >= 100 and red_hue:
            return "deep red"
        return "pink"

    if raw_class == "white":
        if purple_hue and ms >= 20:
            return purple_family(mh, mv)
        if ms > 90:
            return "pale red"
        if ms > 55:
            return "pale pink"
        return "white"

    if raw_class == "indigo_violet":
        if red_hue and ms >= 120 and mv < 160:
            return "deep red"
        if red_hue and ms >= 80:
            return "red"
        if not purple_hue:
            return "red"
        return purple_family(mh, mv)

    if raw_class == "purple":
        if red_hue and ms >= 120 and mv < 160:
            return "deep red"
        if red_hue and ms >= 80:
            return "red"
        if mh < 128:
            return "dark red purple"
        return purple_family(mh, mv)

    return raw_class

# ------------------------------
# Decide final colour based on coating
# ------------------------------
def coating_override(base_colour, coated, ratio, stats):
    if not coated:
        return base_colour
    red_hue = (stats['mh'] <= 15) or (stats['mh'] >= 168)
    p75s = stats['p75s']
    # Special handling for red‑family tongues
    if red_hue and base_colour in ("pale pink", "pale", "pale red", "pink"):
        sat_spread = p75s - stats['ms']
        if sat_spread >= 25 and ratio >= 0.18:
            return "white coated"
        elif ratio >= 0.40:
            return "white coated"
        else:
            return base_colour
    # Default per‑colour threshold
    threshold = COAT_LIMITS.get(base_colour, COAT_THRESHOLD)
    if ratio >= threshold:
        return "white coated"
    return base_colour

# ------------------------------
# Load colour model from checkpoint
# ------------------------------
def load_color_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Colour model not found: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint['class_names']
    img_size = checkpoint.get('img_size', 224)
    backbone = checkpoint.get('backbone', 'tf_efficientnetv2_b2.in1k')
    model = timm.create_model(backbone, pretrained=False, num_classes=len(class_names))
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
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
        'img_size': img_size,
    }

# ------------------------------
# Main prediction function
# ------------------------------
def predict_advanced_color(color_model, pil_img, device):
    """
    Returns:
        final_colour   : e.g. "white coated" (if coating detected) or base colour
        base_colour    : underlying colour from model + HSV refinement
        confidence     : float 0-1 (softmax probability)
        coated         : bool
        coating_ratio  : float 0-1
    """
    # Model inference with softmax
    tensor = color_model['transform'](pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = color_model['model'](tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()
    idx = probs.index(max(probs))
    raw_class = color_model['class_names'][idx]
    confidence = probs[idx]   # 0.0 – 1.0

    # HSV analysis
    img_np = np.array(pil_img.convert('RGB'))
    stats = hsv_stats(img_np)
    coated, ratio = coating_ratio(img_np)

    # Base colour (without coating influence)
    base_colour = refine(raw_class, stats)

    # Final colour (apply coating override)
    final_colour = coating_override(base_colour, coated, ratio, stats)

    return {
        'final_colour':   final_colour,
        'base_colour':    base_colour,
        'confidence':     confidence,
        'coated':         coated,
        'coating_ratio':  round(ratio, 4),
    }
