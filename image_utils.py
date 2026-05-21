import base64
from io import BytesIO
from PIL import Image

def encode_image(pil_img: Image.Image, format: str = "JPEG") -> str:
    buffer = BytesIO()
    if pil_img.mode in ("RGBA", "LA", "P"):
        pil_img = pil_img.convert("RGB")
    pil_img.save(buffer, format=format)
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str
