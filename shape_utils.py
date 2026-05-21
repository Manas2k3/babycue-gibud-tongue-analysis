"""
shape_utils.py
Stub — replace load_shape_cnn / load_shape_ml / predict_shape
with your real implementation.
"""
import os


def load_shape_cnn(model_path=None, device=None):
    """
    Load CNN shape classifier.
    Returns (model, class_names) tuple — both None until implemented.
    """
    if model_path and os.path.exists(model_path):
        # TODO: load real model
        pass
    return None, None


def load_shape_ml(model_path=None):
    """
    Load ML ensemble for shape.
    Returns None until implemented.
    """
    if model_path and os.path.exists(model_path):
        # TODO: load e.g. joblib.load(model_path)
        pass
    return None


def predict_shape(cnn_model, class_names, ml_model, pil_img):
    """
    Predict tongue shape.
    Returns a dict with at least a 'label' key.
    Expected labels: 'normal' | 'oval' | 'rectangle'
    """
    if cnn_model is None and ml_model is None:
        return {"label": "normal", "note": "placeholder — no model loaded"}
    # ------------------------------------------------------------------
    # TODO: implement real inference here
    # ------------------------------------------------------------------
    return {"label": "normal"}
