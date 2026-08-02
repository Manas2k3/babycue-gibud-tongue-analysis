import os
import io
import logging

import torch
from flask import Flask, request, jsonify, render_template_string
from PIL import Image

from color_utils import load_color_model, predict_advanced_color
from texture_utils import load_texture_model, predict_texture
from shape_utils import load_shape_cnn, load_shape_ml, predict_shape
from crack_utils import load_crack_model, predict_crack
from image_utils import encode_image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
logging.basicConfig(level=logging.INFO)

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
)
logging.info(f"Using device: {DEVICE}")

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

def load_all_models():
    logging.info("Loading colour model...")
    models["color"] = load_color_model(_require_path("color"), DEVICE)

    logging.info("Loading texture model...")
    models["texture"] = load_texture_model()

    logging.info("Loading shape CNN...")
    models["shape_cnn"], models["shape_classes"] = load_shape_cnn(
        _require_path("shape_cnn"), DEVICE
    )

    logging.info("Loading shape ML ensemble...")
    models["shape_ml"] = load_shape_ml(_require_path("shape_ml"))

    logging.info("Loading crack model...")
    models["crack"] = load_crack_model(_require_path("crack"), DEVICE)

    logging.info("All models loaded successfully.")

# ------------------------------------------------------------------
# HTML Template – Color Analysis only (no confidence, no overall health)
# ------------------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Tongue Analysis API</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }
        .container { max-width: 900px; margin: 0 auto; }
        .header { text-align: center; color: white; margin-bottom: 40px; }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .upload-section { background: white; border-radius: 10px; padding: 30px; box-shadow: 0 10px 40px rgba(0,0,0,0.2); margin-bottom: 20px; }
        .upload-area { border: 3px dashed #667eea; border-radius: 8px; padding: 40px; text-align: center; cursor: pointer; transition: all 0.3s; }
        .upload-area:hover { border-color: #764ba2; background: #f5f5f5; }
        .upload-area input { display: none; }
        .upload-label { cursor: pointer; display: block; }
        .btn { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px 30px; border: none; border-radius: 5px; cursor: pointer; font-size: 1em; margin-top: 15px; transition: transform 0.2s; }
        .btn:hover { transform: scale(1.05); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: scale(1); }
        .loading { display: none; text-align: center; color: #667eea; margin: 20px 0; }
        .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #667eea; border-radius: 50%; width: 30px; height: 30px; animation: spin 1s linear infinite; margin: 0 auto 10px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .results { background: white; border-radius: 10px; padding: 30px; box-shadow: 0 10px 40px rgba(0,0,0,0.2); }
        .result-hidden { display: none; }
        .result-item { margin: 20px 0; padding: 15px; background: #f9f9f9; border-left: 4px solid #667eea; border-radius: 4px; }
        .result-item h3 { color: #667eea; margin-bottom: 8px; }
        .result-item p { color: #555; }
        .image-preview { margin: 20px 0; text-align: center; }
        .image-preview img { max-width: 300px; border-radius: 8px; box-shadow: 0 5px 15px rgba(0,0,0,0.2); }
        .error { color: #e74c3c; background: #fadbd8; padding: 15px; border-radius: 5px; border-left: 4px solid #e74c3c; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌍 Tongue Analysis</h1>
            <p>Upload an image for AI-powered tongue analysis</p>
        </div>
        
        <div class="upload-section">
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="upload-area" onclick="document.getElementById('imageInput').click()">
                    <label class="upload-label" for="imageInput">
                        <div style="font-size: 2em; margin-bottom: 10px;">📸</div>
                        <div><strong>Click to upload or drag and drop</strong></div>
                        <div style="color: #999; font-size: 0.9em; margin-top: 5px;">JPG, PNG (Max 16MB)</div>
                    </label>
                    <input id="imageInput" type="file" accept="image/*" required />
                </div>
                <button type="submit" class="btn">Analyze Image</button>
            </form>
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing image... Please wait</p>
            </div>
        </div>

        <div class="results result-hidden" id="results">
            <div id="errorMsg" class="error result-hidden"></div>
            <div id="successContent">
                <div class="image-preview">
                    <img id="uploadedImage" src="" alt="Uploaded image" />
                </div>

                <div class="result-item">
                    <h3>🎨 Color Analysis</h3>
                    <p><strong>Final Colour:</strong> <span id="finalColorLabel"></span></p>
                    <p><strong>Base Colour:</strong> <span id="baseColorLabel"></span></p>
                </div>

                <div class="result-item">
                    <h3>✨ Texture</h3>
                    <p><span id="textureLabel"></span></p>
                </div>

                <div class="result-item">
                    <h3>📐 Shape</h3>
                    <p><span id="shapeLabel"></span></p>
                </div>

                <div class="result-item">
                    <h3>🔍 Cracks</h3>
                    <p><span id="crackLabel"></span></p>
                </div>

                <div class="result-item">
                    <h3>🧂 Coating</h3>
                    <p><span id="coatingLabel"></span></p>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('uploadForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const fileInput = document.getElementById('imageInput');
            const file = fileInput.files[0];
            if (!file) return;
            
            const formData = new FormData();
            formData.append('image', file);
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').classList.add('result-hidden');
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                document.getElementById('loading').style.display = 'none';
                
                if (data.status === 'success') {
                    document.getElementById('uploadedImage').src = 'data:image/jpeg;base64,' + data.uploaded_image;
                    document.getElementById('finalColorLabel').textContent = data.color.final_colour || 'N/A';
                    document.getElementById('baseColorLabel').textContent = data.color.base_colour || 'N/A';
                    document.getElementById('textureLabel').textContent = data.texture.label || 'N/A';
                    document.getElementById('shapeLabel').textContent = data.shape.label || 'N/A';
                    document.getElementById('crackLabel').textContent = data.cracks.label || 'N/A';
                    document.getElementById('coatingLabel').textContent = data.coating.label || 'N/A';
                    document.getElementById('results').classList.remove('result-hidden');
                } else {
                    showError(data.message || 'Analysis failed');
                }
            } catch (error) {
                document.getElementById('loading').style.display = 'none';
                showError('Error: ' + error.message);
            }
        });
        
        function showError(message) {
            const errorDiv = document.getElementById('errorMsg');
            errorDiv.textContent = message;
            errorDiv.classList.remove('result-hidden');
            document.getElementById('results').classList.remove('result-hidden');
        }
        
        const uploadArea = document.querySelector('.upload-area');
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => uploadArea.style.background = '#f0f0f0');
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => uploadArea.style.background = '');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            document.getElementById('imageInput').files = files;
        });
    </script>
</body>
</html>
"""

# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": str(DEVICE)}), 200

@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "Empty filename"}), 400

    try:
        pil_img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        return jsonify({"status": "error", "message": f"Invalid image: {e}"}), 400

    # Colour analysis
    try:
        color_result = predict_advanced_color(models["color"], pil_img, DEVICE)
    except Exception as e:
        logging.error(f"Colour model error: {e}")
        color_result = {
            "final_colour": "error", "base_colour": "error",
            "confidence": 0.0, "coated": False, "coating_ratio": 0.0,
            "coating_color": "error",
        }

    # Texture
    try:
        texture_result = predict_texture(models["texture"], pil_img)
    except Exception as e:
        logging.error(f"Texture model error: {e}")
        texture_result = {"label": "error"}

    # Shape
    try:
        shape_result = predict_shape(
            models["shape_cnn"], models["shape_classes"], models["shape_ml"], pil_img
        )
    except Exception as e:
        logging.error(f"Shape model error: {e}")
        shape_result = {"label": "error"}

    # Cracks
    try:
        crack_result = predict_crack(models["crack"], pil_img)
    except Exception as e:
        logging.error(f"Crack model error: {e}")
        crack_result = {"label": "error"}

    # Coating (derived from colour result)
    coating_result = {
        "label": color_result.get("coating_color", "none"),
        "ratio": color_result.get("coating_ratio", 0.0)
    }

    response = {
        "status":           "success",
        "uploaded_image":   encode_image(pil_img),
        "color":            color_result,
        "texture":          texture_result,
        "shape":            shape_result,
        "cracks":           crack_result,
        "coating":          coating_result,
    }
    return jsonify(response), 200

if __name__ == "__main__":
    load_all_models()
    app.run(host="0.0.0.0", port=5000, debug=False)
