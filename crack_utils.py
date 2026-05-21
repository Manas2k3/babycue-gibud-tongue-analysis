"""
crack_utils.py
Stub — replace load_crack_model / predict_crack / detect_coating
with your real implementation.
"""
import os


def load_crack_model(model_path=None, device=None):
    """Load crack detection model. Returns None until implemented."""
    if model_path and os.path.exists(model_path):
        # TODO: load real model
        pass
    return None


def predict_crack(model, pil_img):
    """
    Predict whether tongue is cracked.
    Returns a dict with at least a 'label' key.
    Expected labels: 'cracked' | 'non_cracked'
    """
    if model is None:
        return {"label": "non_cracked", "note": "placeholder — no model loaded"}
    # ------------------------------------------------------------------
    # TODO: implement real inference here
    # ------------------------------------------------------------------
    return {"label": "non_cracked"}


def detect_coating(pil_img):
    """
    Detect tongue coating type.
    Returns a dict with at least a 'label' key.
    Expected labels: 'none' | 'white'
    """
    return {"label": "none", "note": "placeholder — uses color_utils coating_ratio instead"}
