import io
import base64
from PIL import Image

def encode_image(pil_img, max_size=800, quality=85):
    """
    Resize and encode a PIL image to base64 JPEG string.
    
    Args:
        pil_img: PIL Image object
        max_size: Maximum dimension (maintains aspect ratio)
        quality: JPEG quality (1-100)
    
    Returns:
        Base64 encoded string
    """
    img = pil_img.copy()
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def load_image_from_bytes(byte_data):
    """Convert uploaded file bytes to PIL RGB image."""
    return Image.open(io.BytesIO(byte_data)).convert("RGB")
