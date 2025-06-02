import cv2
from PIL import Image, ImageEnhance
import base64
import io

# Fonction d'amélioration de qualité avec Pillow
def improve_image_quality(np_img):
    # Convertir d'abord en format PIL
    image = Image.fromarray(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))

    # Amélioration
    image = ImageEnhance.Sharpness(image).enhance(2.0)
    image = ImageEnhance.Contrast(image).enhance(1.5)
    image = ImageEnhance.Brightness(image).enhance(1.2)

    return image

# Fonction pour convertir une image PIL en base64
def pil_to_base64(pil_img):
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')