from flask import Blueprint, request, jsonify
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io
import os

iris_bp = Blueprint('iris', __name__)

# Simplified iris extraction using OpenCV only (no dlib dependency)
def extract_iris_from_image(image_data):
    try:
        # Convert image data to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Unable to load image")

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use OpenCV's built-in face detector (Haar cascade)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            raise ValueError("No face detected in image")

        # Take the first detected face
        (x, y, w, h) = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) < 2:
            raise ValueError("Could not detect both eyes")

        # Sort eyes by x-coordinate (left to right)
        eyes = sorted(eyes, key=lambda eye: eye[0])

        # Extract left and right eye regions
        left_eye_region = eyes[0]
        right_eye_region = eyes[1]

        # Add margin around eyes
        margin = 20

        # Extract left eye
        ex, ey, ew, eh = left_eye_region
        left_eye = roi_color[max(0, ey-margin):min(roi_color.shape[0], ey+eh+margin),
                            max(0, ex-margin):min(roi_color.shape[1], ex+ew+margin)]

        # Extract right eye
        ex, ey, ew, eh = right_eye_region
        right_eye = roi_color[max(0, ey-margin):min(roi_color.shape[0], ey+eh+margin),
                             max(0, ex-margin):min(roi_color.shape[1], ex+ew+margin)]

        # Check if extracted regions are valid
        if left_eye.size == 0 or right_eye.size == 0:
            raise ValueError("Unable to extract eye regions")

        # Resize for better iris detail visibility
        left_eye = cv2.resize(left_eye, (200, 100))
        right_eye = cv2.resize(right_eye, (200, 100))

        return left_eye, right_eye

    except Exception as e:
        print(f"Error: {e}")
        return None, None

# Fonction d'amélioration de qualité avec Pillow
def improve_image_quality_with_pillow(np_img):
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

@iris_bp.route('/extract-iris', methods=['POST'])
def extract_iris():
    if 'image' not in request.files:
        return jsonify({'error': 'Aucune image fournie'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400
    
    # Lire les données de l'image
    image_data = file.read()
    
    try:
        # Extraire les iris
        left_iris, right_iris = extract_iris_from_image(image_data)
        
        if left_iris is None or right_iris is None:
            return jsonify({'error': 'Échec de l\'extraction des iris'}), 400
        
        # Améliorer la qualité des images
        improved_left = improve_image_quality_with_pillow(left_iris)
        improved_right = improve_image_quality_with_pillow(right_iris)
        
        # Convertir en base64 pour le retour JSON
        left_iris_base64 = pil_to_base64(improved_left)
        right_iris_base64 = pil_to_base64(improved_right)
        
        return jsonify({
            'left_iris': left_iris_base64,
            'right_iris': right_iris_base64
        })
    
    except Exception as e:
        return jsonify({'error': f"Error during extraction: {str(e)}"}), 400

