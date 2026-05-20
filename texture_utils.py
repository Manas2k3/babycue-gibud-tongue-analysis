import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model architecture (must match training)
def _build_texture_model(num_classes=3):
    backbone = models.resnet50(weights=None)
    backbone.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.BatchNorm1d(512),
        nn.GELU(),
        nn.Dropout(p=0.4),
        nn.Linear(512, 256),
        nn.GELU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    return backbone

def load_texture_model():
    model = _build_texture_model()
    state = torch.load("models/best_tongue_model.pth", map_location=DEVICE)
    # Remove possible _orig_mod. prefix from torch.compile
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model

def predict_texture(model, pil_img):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()
    idx = int(torch.argmax(logits, dim=1).item())
    labels = ["normal", "tender", "tough"]
    return {"label": labels[idx], "probabilities": {l: round(p,4) for l,p in zip(labels, probs)}}
