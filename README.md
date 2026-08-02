# 👅 BabyCue GI_GUT: Automated Tongue Analysis Pipeline

A production-ready Deep Learning & Computer Vision pipeline for automated tongue analysis and diagnosis. The pipeline analyzes tongue images across five core diagnostic dimensions: **Color**, **Coating**, **Shape**, **Texture**, and **Cracks**.

It offers both **FastAPI** and **Flask** web services, a **CLI tool** for direct offline analysis, and full **Docker & Cloud Run** containerization.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Model Architecture](#-model-architecture)
- [Repository Structure](#-repository-structure)
- [Prerequisites & Installation](#-prerequisites--installation)
  - [1. Virtual Environment Setup](#1-virtual-environment-setup)
  - [2. PyTorch & Dependencies](#2-pytorch--dependencies)
- [Model Artifacts](#-model-artifacts)
- [Usage & Running Options](#-usage--running-options)
  - [Option A: FastAPI Application (Recommended)](#option-a-fastapi-application-recommended)
  - [Option B: Flask Application](#option-b-flask-application)
  - [Option C: Command Line Interface (CLI)](#option-c-command-line-interface-cli)
  - [Option D: Docker Container Deployment](#option-d-docker-container-deployment)
- [API Reference](#-api-reference)
  - [Health Check](#health-check)
  - [Analyze Tongue Image](#analyze-tongue-image)
- [Response Format](#-response-format)
- [Cloud Deployment (Google Cloud Run)](#-cloud-deployment-google-cloud-run)
- [License](#-license)

---

## 🔬 Overview

Traditional tongue diagnosis assesses multi-dimensional physical attributes of the tongue. This pipeline combines deep convolutional neural networks (EfficientNet-V2, ResNet-50, EfficientNet-B2), Machine Learning ensembles (Random Forest, Gradient Boosting, SVM), and HSV color space computer vision to provide structured diagnostic data from a single uploaded image.

| Attribute | Model Architecture | Diagnostic Classes / Output |
| :--- | :--- | :--- |
| **Color** | EfficientNet-V2 + HSV Color Analysis | `final_colour` (e.g. *"white coated"*), `base_colour` (e.g. *"pale pink"*), `coated` boolean, `coating_ratio` |
| **Coating** | HSV Color Space Segmentation | `label` (*"white"* or *"none"*), `ratio` (0.0 to 1.0) |
| **Shape** | EfficientNet-B2 CNN + ML Ensemble (RF/GBM/SVM) | `label` (*"normal"*, *"oval"*, *"rectangle"*) |
| **Cracks** | ResNet-50 Binary Classifier | `label` (*"cracked"*, *"non_cracked"*) |
| **Texture**| Computer Vision / Texture Feature Extractor | `label` (*"normal"*, *"tender"*, *"tough"*) |

All models load into GPU/CPU memory on application startup to ensure rapid inference per request.

---

## ✨ Key Features

- **Multi-Modal Diagnostic Inference:** Single request triggers inference across 5 specialized diagnostic pipelines.
- **Dual-Layer Color Decomposition:** Computes both overall surface color (`final_colour`) and true underlying tongue tissue color (`base_colour`).
- **Flexible Serving Frameworks:** Includes FastAPI (Async, OpenAPI docs) and Flask implementations.
- **CLI Utility:** Quick offline evaluation of local images without starting a web server.
- **Graceful Error Recovery:** Independent model execution prevents total request failure if one model encounters an anomaly.
- **Container Ready:** Built-in `Dockerfile`, `.dockerignore`, and `.gcloudignore` optimized for Google Cloud Run and Kubernetes.

---

## 📁 Repository Structure

```gfm
GI_GUT/
├── main.py                     # Production FastAPI application (Uvicorn server entrypoint)
├── app.py                      # Flask application server
├── cli.py                      # Command Line Interface (CLI) runner
├── color_utils.py              # EfficientNet-V2 color inference & HSV coating decomposition
├── shape_utils.py              # Hybrid CNN + ML ensemble shape classifier
├── crack_utils.py              # ResNet-50 crack detection module
├── texture_utils.py            # Tongue texture analysis module
├── image_utils.py              # Image encoding & base64 thumbnail generation utilities
├── Dockerfile                  # Containerization specification (Python 3.10-slim + PyTorch CPU)
├── requirements.txt            # Python package dependencies
├── README.md                   # Repository documentation
├── models/                     # Deep learning weights & ensemble artifacts
│   ├── efficientnetv2_tongue_color.pth
│   ├── tongue_cnn_v3.pth
│   ├── tongue_ml_ensemble.pkl
│   ├── cracked_binary_resnet50.pth
│   ├── class_names.json
│   └── README.md
```

---

## ⚙️ Prerequisites & Installation

### 1. Virtual Environment Setup

Python **3.10+** is recommended.

```bash
# Clone the repository
git clone https://github.com/Manas2k3/babycue-gibud-tongue-analysis.git
cd babycue-gibud-tongue-analysis/GI_GUT

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate        # On macOS / Linux
# .venv\Scripts\activate         # On Windows
```

### 2. PyTorch & Dependencies

Install PyTorch according to your environment hardware:

**CPU Version (Standard / Deployment):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**GPU Version (CUDA 11.8+):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Install remaining dependencies:
```bash
pip install -r requirements.txt
```

---

## 🧠 Model Artifacts

Ensure model weight files are present in the `models/` directory:

| Artifact | File Name | Description |
| :--- | :--- | :--- |
| **Color Model** | `models/efficientnetv2_tongue_color.pth` | PyTorch EfficientNet-V2 classifier |
| **Shape CNN** | `models/tongue_cnn_v3.pth` | PyTorch EfficientNet-B2 shape extractor |
| **Shape Ensemble**| `models/tongue_ml_ensemble.pkl` | Scikit-Learn ensemble model |
| **Crack Model** | `models/cracked_binary_resnet50.pth` | PyTorch ResNet-50 binary classifier |
| **Class Names** | `models/class_names.json` | Shape class mappings |

---

## 🚀 Usage & Running Options

### Option A: FastAPI Application (Recommended)

FastAPI provides asynchronous handling and automatic interactive API documentation.

```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```
- **Interactive OpenAPI Documentation:** Open `http://localhost:8080/docs` in your browser.

---

### Option B: Flask Application

To run using the standard Flask development server:

```bash
python app.py
```
*Runs at `http://0.0.0.0:5000` by default.*

For production Flask serving with Waitress (Windows/Cross-platform):
```bash
waitress-serve --port=5000 app:app
```

---

### Option C: Command Line Interface (CLI)

Analyze a local tongue image directly in your terminal:

```bash
# Pretty-printed terminal output
python cli.py path/to/tongue_sample.jpg

# Formatted JSON output
python cli.py path/to/tongue_sample.jpg --json
```

---

### Option D: Docker Container Deployment

Build and run the container locally:

```bash
# Build Docker image
docker build -t tongue-analysis-api .

# Run container on port 8080
docker run -p 8080:8080 tongue-analysis-api
```

---

## 📡 API Reference

### Health Check

Check application status and GPU/CPU hardware acceleration availability.

- **URL:** `GET /health`
- **Response Example:**
  ```json
  {
    "status": "ok",
    "device": "cpu"
  }
  ```

---

### Analyze Tongue Image

Submit a tongue image for multi-dimensional diagnostic analysis.

- **URL:** `POST /analyze`
- **Content-Type:** `multipart/form-data`
- **Form Parameters:**
  - `image` *(File, required)*: Supported extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.

---

## 📊 Response Format

```json
{
  "status": "success",
  "uploaded_image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "color": {
    "final_colour": "white coated",
    "base_colour": "pale pink",
    "confidence": 0.9421,
    "coated": true,
    "coating_ratio": 0.1852
  },
  "texture": {
    "label": "normal"
  },
  "shape": {
    "label": "oval"
  },
  "cracks": {
    "label": "non_cracked"
  },
  "coating": {
    "label": "white",
    "ratio": 0.1852
  }
}
```

### Response Field Descriptions

| Path | Type | Description |
| :--- | :--- | :--- |
| `uploaded_image` | `string` | Base64-encoded thumbnail of the processed image (max 800px width). |
| `color.final_colour` | `string` | Combined tongue color classification (surface layer). |
| `color.base_colour` | `string` | Inferred base tissue color layer (underlying coat). |
| `color.confidence` | `float` | Softmax probability score (0.0 to 1.0). |
| `color.coated` | `boolean` | Flag indicating presence of tongue coating. |
| `color.coating_ratio` | `float` | Ratio of coating surface area relative to total tongue region. |
| `shape.label` | `string` | Morphological classification (`oval`, `rectangle`, `normal`, `error`). |
| `cracks.label` | `string` | Fissure/crack classification (`cracked`, `non_cracked`, `error`). |
| `texture.label` | `string` | Surface texture metric (`normal`, `tender`, `tough`, `error`). |
| `coating.label` | `string` | Coating color/presence (`white`, `none`). |

---

## ☁️ Cloud Deployment (Google Cloud Run)

Deploy directly to Google Cloud Run using the `gcloud` CLI:

```bash
# Build image using Cloud Build
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/tongue-analysis-api

# Deploy service to Cloud Run
gcloud run deploy tongue-analysis-api \
    --image gcr.io/YOUR_PROJECT_ID/tongue-analysis-api \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
