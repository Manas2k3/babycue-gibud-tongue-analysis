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

def load_color_model():
    """Load trained EfficientNet‑V2 colour model."""
    model = timm.create_model('tf_efficientnetv2_b2.in1k', pretrained=False, num_classes=len(COLOR_CLASSES))
    ckpt = torch.load("models/efficientnetv2_tongue_color.pth", map_location=DEVICE)
    if "model_state" in ckpt:
        ckpt = ckpt["model_state"]
    model.load_state_dict(ckpt)
    model.to(DEVICE).eval()
    return model

def refine_color(raw_label, pil_img):
    """
    Apply HSV‑based refinement to colour prediction.
    This is a simplified version – replace with your full logic from colour_testing.ipynb.
    """
    # Convert PIL to numpy for HSV
    img_np = np.array(pil_img.convert("RGB"))
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    mh = np.median(hsv[:,:,0])   # median hue
    ms = np.median(hsv[:,:,1])   # median saturation
    mv = np.median(hsv[:,:,2])   # median value

    # Example rules (customise from your notebook)
    if mv >= 200 and ms < 90:
        return "white" if ms < 40 else "pink"
    if 110 <= mh <= 165 and ms >= 25:
        if mh < 131:
            return "purple"
        else:
            return "indigo_violet"
    # Fallback to raw prediction
    return raw_label

def detect_coating(pil_img, white_v_min=200, white_s_max=40, coat_threshold=0.08, min_blob_area=400):
    """Detect white coating using HSV blob analysis."""
    img_np = np.array(pil_img.convert("RGB"))
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (0, 0, white_v_min), (180, white_s_max, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    _, labels = cv2.connectedComponents(mask)
    blobs = [np.sum(labels == i) for i in range(1, labels.max()+1) if np.sum(labels == i) > min_blob_area]
    ratio = sum(blobs) / (img_np.shape[0] * img_np.shape[1])
    coated = ratio > coat_threshold
    return {"label": "white" if coated else "none"}

def predict_color(model, pil_img):
    """Run colour inference and return label."""
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()
    idx = int(np.argmax(probs))
    raw = COLOR_CLASSES[idx]
    refined = refine_color(raw, pil_img)
    return {"label": refined}
