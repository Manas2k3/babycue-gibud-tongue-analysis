"""
texture_utils.py
Stub — replace the body of load_texture_model / predict_texture
with your real implementation once the model is ready.
"""


def load_texture_model(model_path=None, device=None):
    """Load texture model. Returns None until a real model is provided."""
    return None


def predict_texture(model, pil_img):
    """
    Predict tongue texture.
    Returns a dict with at least a 'label' key.
    Expected labels: 'normal' | 'tender' | 'tough'
    """
    if model is None:
        return {"label": "normal", "note": "placeholder — no model loaded"}
    # ------------------------------------------------------------------
    # TODO: implement real inference here, e.g.:
    #   tensor = transform(pil_img).unsqueeze(0).to(device)
    #   with torch.no_grad():
    #       out = model(tensor)
    #   label = class_names[out.argmax().item()]
    #   return {"label": label}
    # ------------------------------------------------------------------
    return {"label": "normal"}
