from flask import Blueprint, request, jsonify
import cv2
import numpy as np
import dlib
from PIL import Image, ImageEnhance
import base64
import io
import os

iris_bp = Blueprint('iris', __name__)

# Advanced iris extraction using dlib for precise facial landmark detection
def extract_iris_from_image(image_data):
    try:
        # Convert image data to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Unable to load image")

        # Define the path to the shape predictor model
        predictor_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                     "models",
                                     "shape_predictor_68_face_landmarks.dat")

        # Check if the model file exists
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(f"Model file {predictor_path} not found")

        # Initialize dlib's face detector and landmark predictor
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detector(gray)

        if len(faces) == 0:
            raise ValueError("No face detected in image")

        # Take the first detected face
        face = faces[0]

        # Get facial landmarks
        landmarks = predictor(gray, face)

        # Extract eye landmarks (68-point model)
        # Left eye: points 36-41
        # Right eye: points 42-47
        left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

        # Calculate bounding rectangles for eyes
        left_eye_rect = cv2.boundingRect(np.array(left_eye_points))
        right_eye_rect = cv2.boundingRect(np.array(right_eye_points))

        # Add margin around eyes
        margin = 15
        left_eye_x, left_eye_y, left_eye_w, left_eye_h = left_eye_rect
        right_eye_x, right_eye_y, right_eye_w, right_eye_h = right_eye_rect

        # Extract eye regions with margin
        left_eye = frame[max(0, left_eye_y-margin):min(frame.shape[0], left_eye_y+left_eye_h+margin),
                         max(0, left_eye_x-margin):min(frame.shape[1], left_eye_x+left_eye_w+margin)]
        right_eye = frame[max(0, right_eye_y-margin):min(frame.shape[0], right_eye_y+right_eye_h+margin),
                          max(0, right_eye_x-margin):min(frame.shape[1], right_eye_x+right_eye_w+margin)]

        # Check if extracted regions are valid
        if left_eye.size == 0 or right_eye.size == 0:
            raise ValueError("Unable to extract eye regions")

        # Extract iris from each eye
        left_iris = extract_iris_from_eye(left_eye)
        right_iris = extract_iris_from_eye(right_eye)

        return left_iris, right_iris

    except Exception as e:
        print(f"Error: {e}")
        return None, None

def extract_iris_from_eye(eye_image):
    """Extract iris from a single eye image using circle detection"""
    try:
        # Convert to grayscale
        gray_eye = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_eye, (5, 5), 0)

        # Detect circles (iris) using Hough Circle Transform
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=100,
            param1=100,
            param2=30,
            minRadius=10,
            maxRadius=50
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))

            # Get the first circle (iris)
            x, y, r = circles[0, 0]

            # Create a mask for the iris
            mask = np.zeros_like(gray_eye)
            cv2.circle(mask, (x, y), r, 255, -1)

            # Apply mask to get only the iris
            iris = cv2.bitwise_and(eye_image, eye_image, mask=mask)

            # Crop to the iris region
            x_min = max(0, x - r)
            y_min = max(0, y - r)
            x_max = min(eye_image.shape[1], x + r)
            y_max = min(eye_image.shape[0], y + r)

            iris_cropped = iris[y_min:y_max, x_min:x_max]

            # Resize to standard size
            if iris_cropped.size > 0:
                iris_resized = cv2.resize(iris_cropped, (150, 150))
                return iris_resized

        # If iris detection fails, return the whole eye image resized
        return cv2.resize(eye_image, (150, 150))

    except Exception as e:
        print(f"Error extracting iris: {str(e)}")
        return cv2.resize(eye_image, (150, 150))

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
    
    except FileNotFoundError as e:
        return jsonify({
            'error': f"Model file missing: {str(e)}. Please ensure the shape_predictor_68_face_landmarks.dat file is downloaded."
        }), 400
    except Exception as e:
        return jsonify({'error': f"Error during extraction: {str(e)}"}), 400
