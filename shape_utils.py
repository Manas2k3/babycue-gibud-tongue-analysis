import torch
import torch.nn as nn
import joblib
import numpy as np
import cv2
from torchvision import transforms, models
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- CNN Model (EfficientNet‑B2) ----
def load_shape_cnn():
    ckpt = torch.load("models/tongue_cnn_v3.pth", map_location=DEVICE)
    num_classes = len(ckpt["class_names"])
    model = models.efficientnet_b2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE).eval()
    return model, ckpt["class_names"]

# ---- ML Ensemble ----
def load_shape_ml():
    bundle = joblib.load("models/tongue_ml_ensemble.pkl")
    return bundle   # contains "rf", "gbm", "svm", "le", "feat_cols"

# ---- Feature extraction (Otsu threshold) ----
def extract_shape_features(pil_img):
    """Extract handcrafted features from binary mask image."""
    img = np.array(pil_img.convert("L"))
    # Otsu threshold
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 200:
        return None
    H, W = binary.shape
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h if h > 0 else 0
    circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    hull_perim = cv2.arcLength(hull, True)
    convexity = area / hull_area if hull_area > 0 else 0
    solidity = area / (w * h) if w * h > 0 else 0
    roughness = perimeter / hull_perim if hull_perim > 0 else 1
    if len(cnt) >= 5:
        (cx, cy), (ma, Ma), angle = cv2.fitEllipse(cnt)
        eccentricity = np.sqrt(1 - (min(ma, Ma)/max(ma, Ma))**2) if max(ma, Ma) > 0 else 0
    else:
        eccentricity = 0
    moments = cv2.moments(cnt)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    pts = cnt.reshape(-1, 2).astype(np.float32)
    tip_mask = binary[y + int(0.80*h): y + h, x: x + w]
    tip_cols = np.any(tip_mask > 0, axis=0)
    tip_width_r = tip_cols.sum() / w if w > 0 else 0
    base_mask = binary[y: y + int(0.20*h), x: x + w]
    base_cols = np.any(base_mask > 0, axis=0)
    base_width_r = base_cols.sum() / w if w > 0 else 0
    mid = x + w // 2
    left_mass = float(binary[:, x:mid].sum())
    right_mass = float(binary[:, mid: x + w].sum())
    total_mass = left_mass + right_mass
    symmetry = 1 - abs(left_mass - right_mass) / total_mass if total_mass > 0 else 0
    if len(pts) >= 2:
        _, eigvals, _ = cv2.PCACompute2(pts, mean=None)
        elongation = float(eigvals[0, 0] / (eigvals[0, 1] + 1e-6))
    else:
        elongation = 1.0
    indentation = (hull_area - area) / hull_area if hull_area > 0 else 0
    width_img_ratio = w / W if W > 0 else 0
    # Build feature dictionary (must match FEAT_COLS order in training)
    feat = {
        "area": area, "aspect_ratio": aspect_ratio,
        "circularity": circularity, "convexity": convexity,
        "solidity": solidity, "roughness": roughness,
        "eccentricity": eccentricity,
        "tip_width_ratio": tip_width_r, "base_width_ratio": base_width_r,
        "symmetry": symmetry, "elongation": elongation,
        "indentation": indentation, "width_img_ratio": width_img_ratio,
        **{f"hu_{i}": hu_log[i] for i in range(7)},
    }
    return feat

def predict_ml(feat_dict, bundle, top_k=3):
    """Run ML ensemble prediction."""
    le = bundle["le"]
    feat_cols = bundle["feat_cols"]
    x = np.array([[feat_dict.get(c, 0) for c in feat_cols]])
    probs = np.mean([
        bundle["rf"].predict_proba(x),
        bundle["gbm"].predict_proba(x),
        bundle["svm"].predict_proba(x),
    ], axis=0)[0]
    top_idx = np.argsort(probs)[::-1][:top_k]
    return [(le.inverse_transform([i])[0], float(probs[i])) for i in top_idx]

def predict_shape(cnn_model, cnn_classes, ml_bundle, pil_img, ml_weight=0.35):
    """Combine CNN and ML predictions for shape."""
    # ---- CNN prediction ----
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = cnn_model(tensor)
        probs = torch.softmax(out, dim=1).squeeze().cpu().numpy()
    cnn_pred = {cnn_classes[i]: float(probs[i]) for i in range(len(cnn_classes))}

    # ---- ML prediction ----
    feat = extract_shape_features(pil_img)
    if feat is not None:
        ml_preds = predict_ml(feat, ml_bundle, top_k=len(cnn_classes))
        ml_probs = {lbl: p for lbl, p in ml_preds}
    else:
        ml_probs = {cls: 1/len(cnn_classes) for cls in cnn_classes}

    # ---- Combine ----
    combined = {}
    for cls in cnn_classes:
        combined[cls] = ml_weight * ml_probs.get(cls, 0) + (1 - ml_weight) * cnn_pred.get(cls, 0)
    best_label = max(combined, key=combined.get)
    return {"label": best_label}
