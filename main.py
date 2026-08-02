import os
import io
import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from color_utils import load_color_model, predict_advanced_color
from texture_utils import load_texture_model, predict_texture
from shape_utils import load_shape_cnn, load_shape_ml, predict_shape
from crack_utils import load_crack_model, predict_crack
from image_utils import encode_image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device setup
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
)
logger.info(f"Using device: {DEVICE}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

def _first_existing_path(*relative_paths):
    for rel in relative_paths:
        candidate = os.path.join(MODELS_DIR, rel)
        if os.path.isfile(candidate):
            return candidate
    return None

MODEL_PATHS = {
    "color": _first_existing_path(
        "efficientnetv2_tongue_color.pth",
        "efficientnetv2_tongue_color_30epochs.pth",
        "efficientnetv2_tongue_color_50epochs.pth",
    ),
    "shape_cnn": _first_existing_path("tongue_cnn_v3.pth"),
    "shape_ml": _first_existing_path("tongue_ml_ensemble.pkl"),
    "crack": _first_existing_path("cracked_binary_resnet50.pth"),
}

models = {}

def _require_path(key):
    path = MODEL_PATHS.get(key)
    if not path:
        raise FileNotFoundError(
            f"Missing model file for '{key}'. Expected file in {MODELS_DIR}."
        )
    return path

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code: load models
    logger.info("Loading colour model...")
    models["color"] = load_color_model(_require_path("color"), DEVICE)

    logger.info("Loading texture model...")
    models["texture"] = load_texture_model()

    logger.info("Loading shape CNN...")
    models["shape_cnn"], models["shape_classes"] = load_shape_cnn(
        _require_path("shape_cnn"), DEVICE
    )

    logger.info("Loading shape ML ensemble...")
    models["shape_ml"] = load_shape_ml(_require_path("shape_ml"))

    logger.info("Loading crack model...")
    models["crack"] = load_crack_model(_require_path("crack"), DEVICE)

    logger.info("All models loaded successfully.")
    
    yield
    
    # Shutdown code: clean up
    models.clear()

app = FastAPI(
    title="Tongue Analysis API",
    description="FastAPI application for automated tongue diagnosis.",
    version="1.0.0",
    lifespan=lifespan
)

def _process_image(file: UploadFile) -> Image.Image:
    try:
        contents = file.file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        return pil_img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/")
def read_root():
    return {"message": "Tongue Analysis FastAPI is running. See /docs for endpoints."}

@app.get("/health")
def health_check():
    return {"status": "ok", "device": str(DEVICE)}

@app.post("/analyze/color")
async def analyze_color(image: UploadFile = File(...)):
    pil_img = _process_image(image)
    try:
        color_result = predict_advanced_color(models["color"], pil_img, DEVICE)
        return {"status": "success", "color": color_result}
    except Exception as e:
        logger.error(f"Colour model error: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": "Failed to analyze color"})

@app.post("/analyze/texture")
async def analyze_texture(image: UploadFile = File(...)):
    pil_img = _process_image(image)
    try:
        texture_result = predict_texture(models["texture"], pil_img)
        return {"status": "success", "texture": texture_result}
    except Exception as e:
        logger.error(f"Texture model error: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": "Failed to analyze texture"})

@app.post("/analyze/shape")
async def analyze_shape(image: UploadFile = File(...)):
    pil_img = _process_image(image)
    try:
        shape_result = predict_shape(models["shape_cnn"], models["shape_classes"], models["shape_ml"], pil_img)
        return {"status": "success", "shape": shape_result}
    except Exception as e:
        logger.error(f"Shape model error: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": "Failed to analyze shape"})

@app.post("/analyze/cracks")
async def analyze_cracks(image: UploadFile = File(...)):
    pil_img = _process_image(image)
    try:
        crack_result = predict_crack(models["crack"], pil_img)
        return {"status": "success", "cracks": crack_result}
    except Exception as e:
        logger.error(f"Crack model error: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": "Failed to analyze cracks"})

@app.post("/analyze/coating")
async def analyze_coating(image: UploadFile = File(...)):
    pil_img = _process_image(image)
    try:
        color_result = predict_advanced_color(models["color"], pil_img, DEVICE)
        coating_result = {
            "label": "white" if color_result.get("coated", False) else "none",
            "ratio": color_result.get("coating_ratio", 0.0)
        }
        return {"status": "success", "coating": coating_result}
    except Exception as e:
        logger.error(f"Coating extraction error: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": "Failed to analyze coating"})

@app.post("/analyze/all")
async def analyze_all(image: UploadFile = File(...)):
    pil_img = _process_image(image)
    
    # Color
    try:
        color_result = predict_advanced_color(models["color"], pil_img, DEVICE)
    except Exception as e:
        logger.error(f"Colour model error: {e}")
        color_result = {
            "final_colour": "error", "base_colour": "error",
            "confidence": 0.0, "coated": False, "coating_ratio": 0.0,
        }

    # Texture
    try:
        texture_result = predict_texture(models["texture"], pil_img)
    except Exception as e:
        logger.error(f"Texture model error: {e}")
        texture_result = {"label": "error"}

    # Shape
    try:
        shape_result = predict_shape(models["shape_cnn"], models["shape_classes"], models["shape_ml"], pil_img)
    except Exception as e:
        logger.error(f"Shape model error: {e}")
        shape_result = {"label": "error"}

    # Cracks
    try:
        crack_result = predict_crack(models["crack"], pil_img)
    except Exception as e:
        logger.error(f"Crack model error: {e}")
        crack_result = {"label": "error"}

    # Coating
    coating_result = {
        "label": "white" if color_result.get("coated", False) else "none",
        "ratio": color_result.get("coating_ratio", 0.0)
    }

    response = {
        "status": "success",
        "uploaded_image": encode_image(pil_img),
        "color": color_result,
        "texture": texture_result,
        "shape": shape_result,
        "cracks": crack_result,
        "coating": coating_result,
    }
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)
