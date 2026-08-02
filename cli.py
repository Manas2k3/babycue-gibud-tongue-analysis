import argparse
import sys
import os
import json
from PIL import Image

import app  # import models and inference functions from app.py

def analyze_image(image_path, json_output=False):
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        sys.exit(1)
        
    try:
        pil_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error: Invalid image: {e}")
        sys.exit(1)
        
    print("Loading models (this might take a few seconds)...")
    # Suppress verbose logging if we are not outputting json
    if not json_output:
        import logging
        logging.getLogger().setLevel(logging.ERROR)
        
    app.load_all_models()
    
    if not json_output:
        print(f"\nAnalyzing image: {image_path}...\n")
    
    # Colour analysis
    try:
        color_result = app.predict_advanced_color(app.models["color"], pil_img, app.DEVICE)
    except Exception as e:
        if not json_output:
            print(f"Colour model error: {e}")
        color_result = {
            "final_colour": "error", "base_colour": "error",
            "confidence": 0.0, "coated": False, "coating_ratio": 0.0,
        }

    # Texture
    try:
        texture_result = app.predict_texture(app.models["texture"], pil_img)
    except Exception as e:
        if not json_output:
            print(f"Texture model error: {e}")
        texture_result = {"label": "error"}

    # Shape
    try:
        shape_result = app.predict_shape(
            app.models["shape_cnn"], app.models["shape_classes"], app.models["shape_ml"], pil_img
        )
    except Exception as e:
        if not json_output:
            print(f"Shape model error: {e}")
        shape_result = {"label": "error"}

    # Cracks
    try:
        crack_result = app.predict_crack(app.models["crack"], pil_img)
    except Exception as e:
        if not json_output:
            print(f"Crack model error: {e}")
        crack_result = {"label": "error"}

    # Coating (derived from colour result)
    coating_result = {
        "label": "white" if color_result.get("coated", False) else "none",
        "ratio": color_result.get("coating_ratio", 0.0)
    }

    if json_output:
        response = {
            "color": color_result,
            "texture": texture_result,
            "shape": shape_result,
            "cracks": crack_result,
            "coating": coating_result,
        }
        print(json.dumps(response, indent=2))
    else:
        print("--- Analysis Results ---")
        print(f"🎨 Color Analysis:")
        print(f"   Final Colour: {color_result.get('final_colour', 'N/A')}")
        print(f"   Base Colour:  {color_result.get('base_colour', 'N/A')}")
        print(f"✨ Texture: {texture_result.get('label', 'N/A')}")
        print(f"📐 Shape: {shape_result.get('label', 'N/A')}")
        print(f"🔍 Cracks: {crack_result.get('label', 'N/A')}")
        print(f"🧂 Coating: {coating_result.get('label', 'N/A')} (Ratio: {coating_result.get('ratio', 0):.2f})")
        print("------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tongue Analysis CLI")
    parser.add_argument("image_path", help="Path to the tongue image")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    args = parser.parse_args()
    
    analyze_image(args.image_path, args.json)
