import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_crack_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.load("models/cracked_binary_resnet50.pth", map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model

def predict_crack(model, pil_img):
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
    labels = ["cracked", "non_cracked"]
    return {"label": labels[idx], "probabilities": {l: round(p,4) for l,p in zip(labels, probs)}}
