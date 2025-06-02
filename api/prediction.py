from flask import current_app, jsonify, request, Blueprint
import numpy as np
import tensorflow as tf
import io
import base64
from PIL import Image
import firebase_admin
from firebase_admin import credentials, firestore, storage
import os
import uuid
from datetime import datetime

prediction_bp = Blueprint('prediction', __name__)

# Initialize Firebase if not already initialized
if not firebase_admin._apps:
    try:
        # Try to use environment variables first (for Railway deployment)
        firebase_config = {
            "type": os.environ.get('FIREBASE_TYPE', 'service_account'),
            "project_id": os.environ.get('FIREBASE_PROJECT_ID'),
            "private_key_id": os.environ.get('FIREBASE_PRIVATE_KEY_ID'),
            "private_key": os.environ.get('FIREBASE_PRIVATE_KEY', '').replace('\\n', '\n'),
            "client_email": os.environ.get('FIREBASE_CLIENT_EMAIL'),
            "client_id": os.environ.get('FIREBASE_CLIENT_ID'),
            "auth_uri": os.environ.get('FIREBASE_AUTH_URI', 'https://accounts.google.com/o/oauth2/auth'),
            "token_uri": os.environ.get('FIREBASE_TOKEN_URI', 'https://oauth2.googleapis.com/token'),
            "auth_provider_x509_cert_url": os.environ.get('FIREBASE_AUTH_PROVIDER_X509_CERT_URL', 'https://www.googleapis.com/oauth2/v1/certs'),
            "client_x509_cert_url": os.environ.get('FIREBASE_CLIENT_X509_CERT_URL'),
            "universe_domain": os.environ.get('FIREBASE_UNIVERSE_DOMAIN', 'googleapis.com')
        }

        # Check if all required environment variables are set
        if all([firebase_config['project_id'], firebase_config['private_key'], firebase_config['client_email']]):
            cred = credentials.Certificate(firebase_config)
            # Initialize with storage bucket
            firebase_admin.initialize_app(cred, {
                'storageBucket': f"{firebase_config['project_id']}.appspot.com"
            })
            print("✅ Firebase initialized successfully with environment variables")
            print(f"🔥 Project ID: {firebase_config['project_id']}")
            print(f"🔥 Storage Bucket: {firebase_config['project_id']}.appspot.com")
        else:
            # Fallback to credentials file for local development
            cred_path = os.environ.get('FIREBASE_CREDENTIALS', 'firebase-credentials.json')
            if os.path.exists(cred_path):
                cred = credentials.Certificate(cred_path)
                # Try to get project_id from credentials file
                with open(cred_path, 'r') as f:
                    import json
                    cred_data = json.load(f)
                    project_id = cred_data.get('project_id')

                firebase_admin.initialize_app(cred, {
                    'storageBucket': f"{project_id}.appspot.com"
                })
                print("✅ Firebase initialized successfully with credentials file")
                print(f"🔥 Project ID: {project_id}")
            else:
                print("⚠️ Firebase credentials not found - continuing without Firebase")
                print("🔍 Missing environment variables:")
                for key, value in firebase_config.items():
                    if not value:
                        print(f"   - {key.upper()}: {value}")
                print("📝 Required environment variables:")
                print("   - FIREBASE_PROJECT_ID")
                print("   - FIREBASE_PRIVATE_KEY")
                print("   - FIREBASE_CLIENT_EMAIL")
    except Exception as e:
        print(f"❌ Firebase initialization error: {e}")
        # Continue without Firebase - we'll handle errors when trying to use it

def save_processed_image_to_storage(processed_image, user_id, image_type="single"):
    """
    Save processed image to Firebase Storage

    Args:
        processed_image: Processed image array (numpy array)
        user_id: User ID from the request
        image_type: Type of image ("single", "left", "right")

    Returns:
        str: Public URL of the uploaded image or None if failed
    """
    try:
        # Convert numpy array to PIL Image
        if processed_image.dtype != 'uint8':
            # Normalize to 0-255 range if needed
            if processed_image.max() <= 1.0:
                processed_image = (processed_image * 255).astype('uint8')
            else:
                processed_image = processed_image.astype('uint8')

        # Convert to PIL Image
        if len(processed_image.shape) == 2:  # Grayscale
            pil_image = Image.fromarray(processed_image, mode='L')
        else:  # RGB
            pil_image = Image.fromarray(processed_image, mode='RGB')

        # Convert to bytes
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG', quality=85)
        img_byte_arr.seek(0)

        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"processed_images/{user_id}/{timestamp}_{image_type}_{uuid.uuid4().hex[:8]}.jpg"

        # Upload to Firebase Storage
        bucket = storage.bucket()
        blob = bucket.blob(filename)
        blob.upload_from_file(img_byte_arr, content_type='image/jpeg')

        # Make the blob publicly accessible
        blob.make_public()

        return blob.public_url
    except Exception as e:
        print(f"❌ Firebase Storage upload error: {e}")
        return None

def save_to_firebase(user_id, prediction_data, image_urls=None):
    """
    Save prediction results to Firebase Firestore

    Args:
        user_id: User ID from the request
        prediction_data: Prediction results
        image_urls: URLs to the stored processed images (can be string or dict)
    """
    try:
        # Check if Firebase is initialized
        if not firebase_admin._apps:
            print("❌ Firebase not initialized - cannot save to Firestore")
            return False

        print(f"🔥 Attempting to save prediction for user: {user_id}")

        db = firestore.client()

        # Create a copy to avoid modifying the original
        data_to_save = prediction_data.copy()

        # Add metadata to the prediction data
        data_to_save.update({
            'timestamp': datetime.now(),
            'processed_image_urls': image_urls,
            'source': request.headers.get('User-Agent', 'unknown')
        })

        # Generate a unique document ID
        doc_id = f"{user_id}_{uuid.uuid4()}"

        print(f"🔥 Saving to collection 'iris_predictions' with doc_id: {doc_id}")
        print(f"🔥 Data keys: {list(data_to_save.keys())}")

        # Save to Firestore
        db.collection('iris_predictions').document(doc_id).set(data_to_save)

        print(f"✅ Successfully saved prediction to Firestore: {doc_id}")
        return True
    except Exception as e:
        print(f"❌ Firebase save error: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        import traceback
        print(f"❌ Full traceback: {traceback.format_exc()}")
        return False

# Fonction de traitement d'image pour l'extraction d'iris
def process_eye_image_fixed_pupil_center(image_data, target_size=(196, 196)):
    """
    Extrait l'iris à partir des données d'image brutes, détecte son centre.
    La pupille aura TOUJOURS le même centre que l'iris.
    
    Args:
        image_data (bytes): Données brutes de l'image
        target_size (tuple): Taille de l'image de sortie (largeur, hauteur)
        
    Returns:
        np.array: Image iris prétraitée en niveaux de gris sans pupille, ou None si échec
    """
    # Convertir les données d'image en array numpy
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        print("❌ Impossible de décoder l'image")
        return None

    # Redimensionnement pour détection
    initial_resize_dim = (800, 800)
    img_resized = cv2.resize(img, initial_resize_dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Amélioration pour détection iris
    equalized_iris = cv2.equalizeHist(cv2.medianBlur(gray, 5))

    # Détection iris
    iris_circles = cv2.HoughCircles(
        equalized_iris,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=80,
        param1=100,
        param2=25,
        minRadius=40, maxRadius=150
    )
    if iris_circles is None or len(iris_circles[0]) == 0:
        print("❌ Iris non détecté")
        return None

    iris_x, iris_y, iris_r = np.uint16(np.around(iris_circles[0][0]))

    h_resized, w_resized = initial_resize_dim
    if iris_r > min(h_resized, w_resized) / 2 or iris_r < 30:
        print(f"❌ Rayon iris invalide ({iris_r})")
        return None

    # Détection pupille sur canal V HSV, centre forcé sur iris
    hsv_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    v_channel_resized = hsv_resized[:, :, 2]
    pupil_preprocessed_v = cv2.equalizeHist(cv2.medianBlur(v_channel_resized, 7))

    min_pupil_r_hough = int(iris_r * 0.15)
    max_pupil_r_hough = int(iris_r * 0.5)

    pupil_circles = cv2.HoughCircles(
        pupil_preprocessed_v,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=50,
        param2=15,
        minRadius=min_pupil_r_hough, maxRadius=max_pupil_r_hough
    )

    pupil_r_to_use = int((2/3) * iris_r)  # Rayon par défaut
    if pupil_circles is not None and len(pupil_circles[0]) > 0:
        _, _, r_detected = np.uint16(np.around(pupil_circles[0][0]))
        if r_detected <= (2/3) * iris_r:
            pupil_r_to_use = int(r_detected)
        else:
            print("⚠️ Rayon pupille détecté > 2/3 iris. Utilisation rayon par défaut.")
    else:
        print("⚠️ Pupille non détectée. Utilisation rayon par défaut.")

    # Suppression pupille sur image couleur
    processed_img = img_resized.copy()
    mask_iris = np.zeros_like(gray)
    cv2.circle(mask_iris, (iris_x, iris_y), iris_r, 255, -1)
    processed_img = cv2.bitwise_and(processed_img, processed_img, mask=mask_iris)

    cv2.circle(processed_img, (iris_x, iris_y), pupil_r_to_use, (0, 0, 0), -1)

    # Conversion en gris + crop + resize final
    iris_gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)

    crop_radius = int(iris_r * 1.1)
    x_start = max(0, iris_x - crop_radius)
    y_start = max(0, iris_y - crop_radius)
    x_end = min(w_resized, iris_x + crop_radius)
    y_end = min(h_resized, iris_y + crop_radius)

    if y_end <= y_start + 10 or x_end <= x_start + 10:
        print("❌ Zone crop trop petite ou invalide")
        return None

    cropped = iris_gray[y_start:y_end, x_start:x_end]
    final_img = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)

    return final_img

def load_model(model_name=None):
    """
    Charge le modèle de prédiction.
    
    Args:
        model_name (str, optional): Nom spécifique du modèle à charger
    
    Returns:
        Le modèle chargé
    """
    try:
        # Définir les chemins de recherche pour les modèles
        model_dir = os.path.join(current_app.root_path, 'models')
        
        # Si un nom spécifique est fourni, essayer de charger ce modèle
        if model_name:
            model_path = os.path.join(model_dir, model_name)
            print(f"🔍 Tentative de chargement du modèle: {model_path}")
            
            if os.path.exists(model_path):
                print(f"✅ Modèle trouvé: {model_path}")
                return tf.keras.models.load_model(model_path)
            else:
                print(f"⚠️ Modèle {model_name} non trouvé à {model_path}")
                # Continuer pour essayer les modèles par défaut
        
        # Essayer de charger les modèles par défaut
        default_models = [
            "iris_model.h5",
            "iris_model.keras",
            "Efficient_10unfrozelayers.keras",
            "iris_model.joblib"
        ]
        
        for model_file in default_models:
            model_path = os.path.join(model_dir, model_file)
            if os.path.exists(model_path):
                print(f"✅ Modèle par défaut trouvé: {model_path}")
                
                # Charger selon le type de fichier
                if model_file.endswith('.h5') or model_file.endswith('.keras'):
                    return tf.keras.models.load_model(model_path)
                elif model_file.endswith('.joblib'):
                    return joblib.load(model_path)
        
        # Aucun modèle trouvé, afficher les fichiers disponibles pour le débogage
        print(f"⚠️ Aucun modèle trouvé dans {model_dir}")
        if os.path.exists(model_dir):
            files = os.listdir(model_dir)
            print(f"📁 Fichiers disponibles dans le répertoire: {files}")
        else:
            print(f"📁 Le répertoire {model_dir} n'existe pas")
        
        print("⚠️ Retour d'un modèle factice pour le développement.")
        return "dummy_model"
    
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        return None

def preprocess_for_prediction(image, target_size=(196, 196)):
    """
    Prétraite l'image pour la prédiction.
    
    Args:
        image (np.array): Image iris prétraitée
        target_size (tuple): Taille cible pour le modèle
        
    Returns:
        np.array: Image prétraitée prête pour la prédiction
    """
    # Redimensionner si nécessaire
    if image.shape[:2] != target_size:
        image = cv2.resize(image, target_size)
    
    # Normaliser les valeurs de pixels entre 0 et 1
    image = image.astype(np.float32) / 255.0
    
    # Ajouter les dimensions de batch et de canal si nécessaire (pour TensorFlow)
    if len(image.shape) == 2:  # Si image en niveaux de gris
        image = np.expand_dims(image, axis=-1)  # Ajouter dimension de canal
    
    image = np.expand_dims(image, axis=0)  # Ajouter dimension de batch
    
    return image

def make_prediction(model, processed_image):
    """
    Fait une prédiction avec le modèle.
    
    Args:
        model: Le modèle de prédiction
        processed_image (np.array): Image prétraitée
        
    Returns:
        dict: Résultats de la prédiction
    """
    try:
        # Si c'est un modèle factice (pour le développement)
        if model == "dummy_model":
            return {
                "message": "Modèle non chargé. Prédiction factice retournée.",
                "prediction": "classe_inconnue",
                "confidence": 0.0
            }
        
        # Faire la prédiction
        if isinstance(model, tf.keras.Model):
            # Modèle TensorFlow
            predictions = model.predict(processed_image)
            
            # Obtenir la classe prédite et la confiance
            if predictions.shape[1] > 1:  # Classification multi-classes
                predicted_class = int(np.argmax(predictions[0]))
                confidence = float(predictions[0][predicted_class])
            else:  # Classification binaire
                predicted_class = 1 if predictions[0][0] > 0.5 else 0
                confidence = float(predictions[0][0]) if predicted_class == 1 else 1 - float(predictions[0][0])
            
            # Obtenir les noms de classes
            try:
                class_names = current_app.class_names
            except AttributeError:
                # Si les noms de classes ne sont pas définis, utiliser les indices
                class_names = [f"classe_{i}" for i in range(len(predictions[0]))]
            
            # Créer un dictionnaire avec les classes et leurs probabilités
            class_predictions = {}
            for i, class_name in enumerate(class_names):
                class_predictions[class_name] = float(predictions[0][i])
            
            return {
                "prediction": str(class_names[predicted_class]),
                "confidence": confidence,
                "class_predictions": class_predictions
            }
        else:
            # Modèle scikit-learn
            # Adapter l'image pour scikit-learn (aplatir)
            flat_image = processed_image.reshape(1, -1)
            predicted_class = model.predict(flat_image)[0]
            
            # Obtenir les probabilités si disponibles
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(flat_image)[0]
                confidence = float(probas[predicted_class])
                
                # Obtenir les noms de classes
                try:
                    class_names = current_app.class_names
                except AttributeError:
                    # Si les noms de classes ne sont pas définis, utiliser les indices
                    class_names = [f"classe_{i}" for i in range(len(probas))]
                
                # Créer un dictionnaire avec les classes et leurs probabilités
                class_predictions = {}
                for i, class_name in enumerate(class_names):
                    class_predictions[class_name] = float(probas[i])
                
                return {
                    "prediction": str(class_names[predicted_class]),
                    "confidence": confidence,
                    "class_predictions": class_predictions
                }
            else:
                confidence = None
                return {
                    "prediction": str(predicted_class),
                    "confidence": confidence
                }
    
    except Exception as e:
        print(f"❌ Erreur lors de la prédiction: {e}")
        return {"error": f"Erreur lors de la prédiction: {str(e)}"}

@prediction_bp.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for making a prediction from an image.
    
    Accepts an image via POST request and returns prediction results.
    Also saves results to Firebase if user_id is provided.
    """
    try:
        # Check if file was uploaded
        if 'image' not in request.files:
            # Check if it's JSON data with base64 image
            if request.is_json and 'image_data' in request.json:
                # Handle base64 encoded image
                import base64
                image_data = base64.b64decode(request.json['image_data'])
                user_id = request.json.get('user_id')
            else:
                return jsonify({'error': 'No image provided. Send as file upload or base64 in JSON'}), 400
        else:
            # Handle file upload
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Read image data
            image_data = file.read()
            user_id = request.form.get('user_id')
        
        # Process the image
        processed_image = process_eye_image_fixed_pupil_center(image_data)
        if processed_image is None:
            return jsonify({'error': 'Failed to process image. Could not detect iris.'}), 400
        
        # Load model (lazy loading)
        if not hasattr(current_app, 'model'):
            current_app.model = load_model()
        
        if current_app.model is None:
            return jsonify({'error': 'Error loading model'}), 500
        
        # Preprocess image for prediction
        prediction_ready_image = preprocess_for_prediction(processed_image)
        
        # Make prediction
        prediction_results = make_prediction(current_app.model, prediction_ready_image)
        
        # Check for errors
        if 'error' in prediction_results:
            return jsonify({'error': prediction_results['error']}), 500
        
        # Save to Firebase if user_id is provided
        if user_id:
            # Add user_id to prediction results
            prediction_results['user_id'] = user_id

            # Save processed image to Firebase Storage
            image_url = save_processed_image_to_storage(processed_image, user_id, "single")

            # Save to Firebase
            firebase_success = save_to_firebase(user_id, prediction_results, image_url)
            prediction_results['saved_to_firebase'] = firebase_success
            if image_url:
                prediction_results['processed_image_url'] = image_url
        
        # Return results
        return jsonify(prediction_results)
    
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@prediction_bp.route('/predict-efficient', methods=['POST'])
def predict_efficient():
    """
    Endpoint pour faire une prédiction avec le modèle EfficientNet.

    Accepte deux images via une requête POST multipart/form-data.
    Traite les deux images, fait une prédiction pour chacune, calcule la moyenne
    et retourne un résultat unique.
    Also saves results to Firebase if user_id is provided.

    Returns:
        JSON: Résultats de la prédiction moyenne ou message d'erreur
    """
    try:
        # Vérifier si les images ont été envoyées
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({'error': 'Deux images sont requises (image1 et image2)'}), 400

        file1 = request.files['image1']
        file2 = request.files['image2']

        # Get user_id from form data
        user_id = request.form.get('user_id')

        if file1.filename == '' or file2.filename == '':
            return jsonify({'error': 'Aucun fichier sélectionné pour une ou les deux images'}), 400
        
        # Vérifier l'extension des fichiers
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        file1_ext = os.path.splitext(file1.filename.lower())[1]
        file2_ext = os.path.splitext(file2.filename.lower())[1]
        
        if file1_ext not in allowed_extensions or file2_ext not in allowed_extensions:
            return jsonify({'error': f'Format de fichier non supporté. Formats acceptés: {", ".join(allowed_extensions)}'}), 400
        
        # Fonction pour traiter une image et obtenir les prédictions
        def process_and_predict(file, image_type="unknown"):
            # Lire les données de l'image et convertir en image PIL
            image_data = file.read()
            image = Image.open(io.BytesIO(image_data))

            # Prétraiter l'image pour EfficientNet (redimensionner à la taille attendue)
            target_size = (224, 224)  # Taille standard pour EfficientNet
            image = image.resize(target_size)

            # Convertir en array pour la prédiction
            img_array = np.array(image)

            # Normaliser l'image si nécessaire (valeurs entre 0 et 1)
            if img_array.max() > 1.0:
                img_array = img_array / 255.0

            # Store processed image for Firebase upload (before adding batch dimension)
            processed_image_for_storage = img_array.copy()

            # Ajouter la dimension de batch
            img_array = np.expand_dims(img_array, 0)

            # Charger le modèle EfficientNet si ce n'est pas déjà fait
            model_name = "Efficient_10unfrozelayers.keras"
            if not hasattr(current_app, 'efficient_model'):
                current_app.efficient_model = load_model(model_name)

            # Check if we got a dummy model (string) instead of a real model
            if isinstance(current_app.efficient_model, str) and current_app.efficient_model == "dummy_model":
                # Return dummy predictions for development/testing
                # Create a dummy array with 8 classes (matching your class_names)
                dummy_predictions = np.zeros(8)
                dummy_predictions[0] = 0.7  # Set highest probability to first class
                return dummy_predictions, processed_image_for_storage

            if current_app.efficient_model is None:
                raise ValueError(f'Erreur lors du chargement du modèle {model_name}')

            # Faire la prédiction
            predictions = current_app.efficient_model.predict(img_array)

            return predictions[0], processed_image_for_storage
        
        # Traiter les deux images et obtenir les prédictions
        file1.seek(0)  # Remettre le pointeur de fichier au début
        file2.seek(0)
        pred1, processed_img1 = process_and_predict(file1, "left")
        file2.seek(0)
        pred2, processed_img2 = process_and_predict(file2, "right")
        
        # Calculer la moyenne des prédictions
        avg_predictions = (pred1 + pred2) / 2.0
        
        # Obtenir les noms de classes
        try:
            class_names = current_app.class_names
        except AttributeError:
            # Si les noms de classes ne sont pas définis, utiliser les indices
            class_names = [f"classe_{i}" for i in range(len(avg_predictions))]
        
        # Trouver la classe avec la plus haute probabilité
        predicted_class_index = np.argmax(avg_predictions)
        predicted_class = class_names[predicted_class_index]
        confidence = float(avg_predictions[predicted_class_index])
        
        # Créer un dictionnaire avec les classes et leurs probabilités moyennes
        class_predictions = {}
        for i, class_name in enumerate(class_names):
            class_predictions[class_name] = float(avg_predictions[i])

        # Prepare prediction results
        prediction_results = {
            "prediction": predicted_class,
            "confidence": confidence,
            "class_predictions": class_predictions,
            "is_dummy_prediction": isinstance(current_app.efficient_model, str),
            "model_type": "efficient_dual_image"
        }

        # Save to Firebase if user_id is provided
        if user_id:
            # Add user_id to prediction results
            prediction_results['user_id'] = user_id

            # Save processed images to Firebase Storage
            image_urls = {}
            left_url = save_processed_image_to_storage(processed_img1, user_id, "left")
            right_url = save_processed_image_to_storage(processed_img2, user_id, "right")

            if left_url:
                image_urls['left'] = left_url
            if right_url:
                image_urls['right'] = right_url

            # Save to Firebase
            firebase_success = save_to_firebase(user_id, prediction_results, image_urls)
            prediction_results['saved_to_firebase'] = firebase_success
            if image_urls:
                prediction_results['processed_image_urls'] = image_urls

        # Retourner les résultats
        return jsonify(prediction_results)

    except Exception as e:
        return jsonify({'error': f'Erreur inattendue: {str(e)}'}), 500

@prediction_bp.route('/test-firebase', methods=['GET'])
def test_firebase():
    """
    Test Firebase connectivity and configuration
    """
    try:
        # Check Firebase initialization
        if not firebase_admin._apps:
            return jsonify({
                'firebase_initialized': False,
                'error': 'Firebase not initialized',
                'message': 'Check environment variables or credentials file'
            }), 500

        # Test Firestore connection
        try:
            db = firestore.client()
            # Try to write a test document
            test_doc = {
                'test': True,
                'timestamp': datetime.now(),
                'message': 'Firebase connectivity test'
            }
            doc_ref = db.collection('test').document('connectivity_test')
            doc_ref.set(test_doc)

            # Try to read it back
            doc = doc_ref.get()
            firestore_status = 'connected' if doc.exists else 'write_failed'

            # Clean up test document
            doc_ref.delete()

        except Exception as e:
            firestore_status = f'error: {str(e)}'

        # Test Storage connection
        try:
            bucket = storage.bucket()
            storage_status = f'connected to bucket: {bucket.name}'
        except Exception as e:
            storage_status = f'error: {str(e)}'

        return jsonify({
            'firebase_initialized': True,
            'firestore_status': firestore_status,
            'storage_status': storage_status,
            'app_name': firebase_admin._apps['[DEFAULT]'].name if '[DEFAULT]' in firebase_admin._apps else 'unknown'
        })

    except Exception as e:
        return jsonify({
            'error': f'Firebase test failed: {str(e)}',
            'firebase_initialized': bool(firebase_admin._apps)
        }), 500


@prediction_bp.route('/analyze-iris-enhanced', methods=['POST'])
def analyze_iris_enhanced():
    """
    Enhanced iris analysis endpoint that accepts:
    - Two iris images (left and right)
    - User profile data
    - Processes both images through ML model
    - Saves complete results to Firestore
    - Returns only primary personality class
    """
    try:
        # Check if both iris images are provided
        if 'left_iris' not in request.files or 'right_iris' not in request.files:
            return jsonify({'error': 'Both left_iris and right_iris images are required'}), 400

        left_iris_file = request.files['left_iris']
        right_iris_file = request.files['right_iris']

        if left_iris_file.filename == '' or right_iris_file.filename == '':
            return jsonify({'error': 'No file selected for one or both iris images'}), 400

        # Get user profile data from form fields
        user_profile = {
            'name': request.form.get('name', ''),
            'email': request.form.get('email', ''),
            'age': request.form.get('age', ''),
            'gender': request.form.get('gender', ''),
            'comments': request.form.get('comments', ''),
            'user_id': request.form.get('user_id', '')
        }

        # Validate required fields
        if not user_profile['email']:
            return jsonify({'error': 'Email is required'}), 400

        # Process both iris images and get predictions
        left_predictions = process_iris_image(left_iris_file)
        right_predictions = process_iris_image(right_iris_file)

        # Calculate combined predictions and confidence scores
        analysis_results = calculate_combined_analysis(left_predictions, right_predictions)

        # Determine primary personality class
        primary_class = analysis_results['primary_class']

        # Convert images to base64 for storage
        left_iris_file.seek(0)
        right_iris_file.seek(0)
        left_iris_base64 = base64.b64encode(left_iris_file.read()).decode('utf-8')
        right_iris_base64 = base64.b64encode(right_iris_file.read()).decode('utf-8')

        # Prepare complete data for Firestore
        firestore_data = {
            'user_profile': user_profile,
            'analysis_results': analysis_results,
            'primary_class': primary_class,
            'left_iris_image': left_iris_base64,
            'right_iris_image': right_iris_base64,
            'analysis_timestamp': datetime.now().isoformat(),
            'source': 'flutter_mobile',
            'version': '2.0'
        }

        # Save to Firestore
        firestore_success = save_iris_analysis_to_firestore(firestore_data)

        # Return only the primary personality class to frontend
        response_data = {
            'primary_class': primary_class,
            'saved_to_firestore': firestore_success
        }

        if not firestore_success:
            response_data['warning'] = 'Analysis completed but failed to save to database'

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500


def process_iris_image(iris_file):
    """Process a single iris image and return predictions"""
    try:
        # Read and preprocess the image
        image_data = iris_file.read()
        image = Image.open(io.BytesIO(image_data))

        # Resize to model input size
        target_size = (224, 224)
        image = image.resize(target_size)

        # Convert to array and normalize
        img_array = np.array(image)
        if img_array.max() > 1.0:
            img_array = img_array / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, 0)

        # Load model if not already loaded
        model_name = "Efficient_10unfrozelayers.keras"
        if not hasattr(current_app, 'efficient_model'):
            current_app.efficient_model = load_model(model_name)

        if current_app.efficient_model is None:
            raise ValueError(f'Error loading model {model_name}')

        # Make prediction
        predictions = current_app.efficient_model.predict(img_array)

        return predictions[0]

    except Exception as e:
        raise Exception(f'Error processing iris image: {str(e)}')


def calculate_combined_analysis(left_predictions, right_predictions):
    """Calculate combined analysis from both iris predictions"""
    try:
        # Get class names
        class_names = getattr(current_app, 'class_names', [f"class_{i}" for i in range(len(left_predictions))])

        # Calculate average predictions
        avg_predictions = (left_predictions + right_predictions) / 2.0

        # Create confidence scores for each class
        confidence_scores = {}
        for i, class_name in enumerate(class_names):
            confidence_scores[class_name] = {
                'left_confidence': float(left_predictions[i]),
                'right_confidence': float(right_predictions[i]),
                'combined_confidence': float(avg_predictions[i])
            }

        # Determine primary class
        primary_class_index = np.argmax(avg_predictions)
        primary_class = class_names[primary_class_index]
        primary_confidence = float(avg_predictions[primary_class_index])

        return {
            'primary_class': primary_class,
            'primary_confidence': primary_confidence,
            'confidence_scores': confidence_scores,
            'left_predictions': left_predictions.tolist(),
            'right_predictions': right_predictions.tolist(),
            'combined_predictions': avg_predictions.tolist()
        }

    except Exception as e:
        raise Exception(f'Error calculating combined analysis: {str(e)}')


def save_iris_analysis_to_firestore(data):
    """Save iris analysis results to Firestore"""
    try:
        db = firestore.client()

        # Generate unique document ID
        doc_id = f"iris_analysis_{data['user_profile']['user_id']}_{uuid.uuid4()}"

        # Save to iris_analysis_results collection
        db.collection('iris_analysis_results').document(doc_id).set(data)

        return True

    except Exception as e:
        print(f"❌ Firestore save error: {e}")
        return False










