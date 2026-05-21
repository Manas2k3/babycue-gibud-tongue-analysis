# Tongue Analysis Flask API

A production‑ready Flask API for automated tongue diagnosis using five complementary deep learning / computer vision models. The API accepts a tongue image and returns predictions for **colour**, **texture**, **shape**, **cracks**, and **coating**.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
  - [Virtual Environment](#virtual-environment)
  - [Dependencies](#dependencies)
- [Model Files](#model-files)
- [Running the API](#running-the-api)
- [API Endpoints](#api-endpoints)
  - [Health Check](#health-check)
  - [Analyse Image](#analyse-image)
- [Response Format](#response-format)
- [Error Handling](#error-handling)
- [Testing](#testing)
- [Deployment](#deployment)
- [Customisation](#customisation)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

This API combines five specialised models to analyse tongue images:

| Task      | Model                                    | Output classes                                      |
|-----------|------------------------------------------|-----------------------------------------------------|
| Colour    | EfficientNet‑V2 + HSV refinement         | final_colour (e.g. "white coated"), base_colour (e.g. "pale pink") |
| Texture   | ResNet‑50 (or stub)                      | normal, tender, tough                               |
| Shape     | CNN + ML ensemble                        | normal, oval, rectangle                             |
| Cracks    | ResNet‑50 binary                         | cracked, non_cracked                                |
| Coating   | HSV blob detection                       | white, none                                         |

All models are loaded once at startup. The API returns a unified JSON response.

## Features

- Multi‑modal tongue analysis
- Base64 thumbnail of uploaded image included in response
- Colour analysis provides both **final colour** (with coating) and **base colour** (underlying tongue colour)
- Graceful error handling – partial results returned if one model fails
- Easy to extend or replace individual models

## Project Structure

```
tongue_api/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── models/                     # Trained model files (not included in repo)
│   ├── efficientnetv2_tongue_color_50epochs.pth
│   ├── tongue_cnn_v3.pth
│   ├── tongue_ml_ensemble.pkl
│   └── cracked_binary_resnet5.pth
└── utils/                      # Helper modules
    ├── __init__.py
    ├── color_utils.py
    ├── texture_utils.py
    ├── shape_utils.py
    ├── crack_utils.py
    └── image_utils.py
```

## Requirements

- Python 3.10 or higher
- pip

## Installation

### Virtual Environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\activate             # Windows
```

### Dependencies

Install PyTorch first (choose CPU or GPU version):

**CPU version** (works on any machine):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**GPU version (CUDA 11.8)**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then install the remaining packages:

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```
flask==2.3.3
timm==0.9.2
opencv-python-headless==4.8.1.78
pillow==10.0.0
numpy==1.24.3
scikit-learn==1.3.0
joblib==1.3.2
```

## Model Files

Place the trained model files inside the `models/` folder. The API expects the following filenames (adjust paths in `app.py` if different):

| Filename                              | Description                     |
|---------------------------------------|---------------------------------|
| `efficientnetv2_tongue_color_50epochs.pth` | Colour EfficientNet‑V2     |
| `tongue_cnn_v3.pth`                   | Shape CNN (EfficientNet‑B2)     |
| `tongue_ml_ensemble.pkl`              | Shape ML ensemble (RF+GBM+SVM)  |
| `cracked_binary_resnet5.pth`          | Cracks ResNet‑5 binary          |

> **Note:** Texture model is currently a stub returning `"normal"`. Replace `texture_utils.load_texture_model()` with your own trained model.

If a file is missing, the corresponding prediction will return `{"label": "error"}`.

## Running the API

```bash
python app.py
```

The server will start at `http://0.0.0.0:5000`. Use `Ctrl+C` to stop.

## API Endpoints

### Health Check

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "ok",
  "device": "cpu"
}
```

### Analyse Image

**Endpoint:** `POST /analyze`

**Request:** `multipart/form-data` with field `image` (upload a tongue image – supported formats: jpg, jpeg, png, bmp, webp).

**Response:** JSON object with all predictions.

## Response Format

```json
{
  "status": "success",
  "uploaded_image": "base64_encoded_thumbnail",
  "color": {
    "final_colour": "white coated",
    "base_colour": "pale pink",
    "confidence": 0.6149,
    "coated": true,
    "coating_ratio": 0.1234
  },
  "texture": { "label": "normal" },
  "shape": { "label": "oval" },
  "cracks": { "label": "non_cracked" },
  "coating": { "label": "white", "ratio": 0.1234 }
}
```

| Field               | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `uploaded_image`    | Base64‑encoded JPEG thumbnail (max 800px) of the uploaded image.            |
| `color.final_colour`| Colour including coating, e.g. `"white coated"`.                            |
| `color.base_colour` | Underlying tongue colour without coating, e.g. `"pale pink"`.               |
| `color.confidence`  | Softmax probability (0–1) from the colour model.                            |
| `color.coated`      | Boolean indicating if coating was detected.                                 |
| `color.coating_ratio` | Fraction of tongue area covered by coating (0–1).                         |
| `texture.label`     | One of `"normal"`, `"tender"`, `"tough"`, or `"error"`.                     |
| `shape.label`       | One of `"normal"`, `"oval"`, `"rectangle"`, or `"error"`.                   |
| `cracks.label`      | One of `"cracked"`, `"non_cracked"`, or `"error"`.                          |
| `coating.label`     | `"white"` if coating detected, else `"none"`.                               |
| `coating.ratio`     | Same as `color.coating_ratio`.                                              |

> **Note:** The `confidence_score` and overall health assessment have been removed from the API. Only per‑attribute results are returned.

## Error Handling

If any model fails to load or raises an exception during inference, the API returns `{"label": "error"}` for that attribute. Other attributes are still processed. The overall request will still return `"status": "success"` as long as the image was read successfully.

## Testing

### Using cURL

```bash
curl -X POST -F "image=@/path/to/tongue.jpg" http://localhost:5000/analyze
```

### Using Python

```python
import requests

with open("tongue.jpg", "rb") as f:
    files = {"image": f}
    resp = requests.post("http://localhost:5000/analyze", files=files)
    print(resp.json())
```

## Deployment

For production, use a WSGI server like **gunicorn** (Linux) or **waitress** (Windows).

**Example with gunicorn:**
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

**Example with waitress (Windows):**
```bash
waitress-serve --port=5000 app:app
```

Ensure `debug=False` in `app.py` (already set).

## Customisation

- **Colour refinement** – Edit `refine()` and `coating_override()` in `utils/color_utils.py` to adjust HSV post‑processing rules.
- **Model paths** – Modify the `MODEL_PATHS` dictionary in `app.py` if your models are stored elsewhere.
- **Remove coating detection** – Change `coated` to always `False` inside `predict_advanced_color()`.
- **Add confidence back** – The backend still computes `color.confidence`; you can re‑expose it in the UI by editing the HTML template.

## Troubleshooting

| Error                                      | Likely cause                                  | Solution                                                                 |
|--------------------------------------------|-----------------------------------------------|--------------------------------------------------------------------------|
| `FileNotFoundError` for a `.pth` file      | Model file missing or wrong filename          | Place correct file in `models/` and ensure name matches `MODEL_PATHS`.   |
| `ModuleNotFoundError: No module named 'utils'` | `utils/__init__.py` missing               | Create an empty `__init__.py` inside the `utils/` folder.                |
| `CUDA out of memory`                       | GPU memory insufficient                       | Force CPU by changing `DEVICE = torch.device("cpu")` in `app.py`.        |
| Slow inference on CPU                      | CPU only                                      | Use GPU if available, or reduce input size in `image_utils.encode_image`.|
| Colour confidence > 100%                   | Old model without softmax                     | Update `color_utils.py` to use `torch.softmax` (already fixed in provided version). |

## License

MIT – free for academic and commercial use.
```

This README matches your current implementation (no overall health score, colour shows final+base, confidence only in JSON, proper file names). Copy and replace your existing `README.md`.
