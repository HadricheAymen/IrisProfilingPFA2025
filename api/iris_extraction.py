from flask import Blueprint, request, jsonify
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io
import dlib

iris_bp = Blueprint('iris', __name__)

# Fonction d'extraction d'iris avec dlib
def extract_iris_from_image(image_data):
    try:
        # Convertir les données d'image en array numpy
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise ValueError("Impossible de charger l'image")

        # Définir le chemin du fichier shape_predictor
        import os
        predictor_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     "models", 
                                     "shape_predictor_68_face_landmarks.dat")
        
        # Vérifier si le fichier existe
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(f"Le fichier {predictor_path} n'existe pas")

        # Initialiser le détecteur de visage et le prédicteur de points de repère de dlib
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        
        # Détecter les visages
        faces = detector(frame)
        
        if len(faces) == 0:
            raise ValueError("Aucun visage détecté dans l'image")
            
        # Prendre le premier visage détecté
        face = faces[0]
        
        # Obtenir les points de repère du visage
        landmarks = predictor(frame, face)
        
        # Points de repère pour les yeux (selon le modèle à 68 points)
        # Œil gauche: points 36-41
        # Œil droit: points 42-47
        left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
        
        # Calculer les rectangles englobants pour les yeux
        left_eye_rect = cv2.boundingRect(np.array(left_eye_points))
        right_eye_rect = cv2.boundingRect(np.array(right_eye_points))
        
        # Ajouter une marge autour des yeux
        margin = 10
        left_eye_x, left_eye_y, left_eye_w, left_eye_h = left_eye_rect
        right_eye_x, right_eye_y, right_eye_w, right_eye_h = right_eye_rect
        
        # Extraire les régions des yeux avec la marge
        left_eye = frame[left_eye_y-margin:left_eye_y+left_eye_h+margin, 
                         left_eye_x-margin:left_eye_x+left_eye_w+margin]
        right_eye = frame[right_eye_y-margin:right_eye_y+right_eye_h+margin, 
                          right_eye_x-margin:right_eye_x+right_eye_w+margin]
        
        # Vérifier que les régions extraites ne sont pas vides
        if left_eye.size == 0 or right_eye.size == 0:
            raise ValueError("Impossible d'extraire les régions des yeux")
        
        # Redimensionner pour mieux voir les détails de l'iris
        left_eye = cv2.resize(left_eye, None, fx=5, fy=5)
        right_eye = cv2.resize(right_eye, None, fx=5, fy=5)
        
        return left_eye, right_eye

    except Exception as e:
        print(f"Erreur : {e}")
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
    
    except FileNotFoundError as e:
        return jsonify({
            'error': f"Fichier de modèle manquant: {str(e)}. Veuillez télécharger le fichier shape_predictor_68_face_landmarks.dat et le placer dans le dossier 'models'."
        }), 400
    except Exception as e:
        return jsonify({'error': f"Erreur lors de l'extraction: {str(e)}"}), 400

