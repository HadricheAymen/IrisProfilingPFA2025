from flask import current_app, jsonify, request, Blueprint
import numpy as np
import tensorflow as tf
import cv2
import io
import base64
from PIL import Image
import firebase_admin
from firebase_admin import credentials, firestore, storage
import os
import uuid
from datetime import datetime
import joblib

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
                print("🔗 Get these from Firebase Console > Project Settings > Service Accounts")
                print("🔗 Generate new private key and extract values from the JSON file")
    except Exception as e:
        print(f"❌ Firebase initialization error: {e}")
        # Continue without Firebase - we'll handle errors when trying to use it

def convert_image_to_base64(processed_image, image_type="single"):
    """
    Convert processed image to base64 string for Firestore storage

    Args:
        processed_image: Processed image array (numpy array)
        image_type: Type of image ("single", "left", "right")

    Returns:
        dict: Dictionary containing base64 image data and metadata, or None if failed
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
        pil_image.save(img_byte_arr, format='JPEG', quality=75)  # Reduced quality for smaller size
        img_byte_arr.seek(0)

        # Convert to base64
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

        # Generate metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_id = f"{timestamp}_{image_type}_{uuid.uuid4().hex[:8]}"

        return {
            'image_data': img_base64,
            'image_type': image_type,
            'image_id': image_id,
            'format': 'JPEG',
            'quality': 75,
            'size_bytes': len(img_base64),
            'dimensions': {
                'width': pil_image.width,
                'height': pil_image.height
            },
            'timestamp': datetime.now()
        }
    except Exception as e:
        print(f"❌ Image to base64 conversion error: {e}")
        return None

def save_to_firebase(user_id, prediction_data, image_data=None):
    """
    Save prediction results to Firebase Firestore with embedded image data

    Args:
        user_id: User ID from the request
        prediction_data: Prediction results
        image_data: Base64 encoded image data (dict or list of dicts)
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
            'processed_images': image_data,  # Store image data directly in Firestore
            'source': request.headers.get('User-Agent', 'unknown')
        })

        # Generate a unique document ID
        doc_id = f"{user_id}_{uuid.uuid4()}"

        print(f"🔥 Saving to collection 'iris_predictions' with doc_id: {doc_id}")
        print(f"🔥 Data keys: {list(data_to_save.keys())}")

        # Ensure all float values are properly preserved
        def preserve_float_precision(obj):
            """Recursively ensure float values maintain their precision"""
            if isinstance(obj, dict):
                return {k: preserve_float_precision(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [preserve_float_precision(item) for item in obj]
            elif isinstance(obj, np.float64) or isinstance(obj, np.float32):
                return float(obj)  # Convert numpy floats to Python floats
            elif isinstance(obj, float):
                return obj  # Keep Python floats as-is
            else:
                return obj

        # Apply float precision preservation
        data_to_save = preserve_float_precision(data_to_save)

        # Check document size (Firestore limit is 1MB) - use a safer size calculation
        try:
            import json
            # Use a custom serializer that handles datetime objects
            def json_serializer(obj):
                if hasattr(obj, 'isoformat'):  # datetime objects
                    return obj.isoformat()
                elif isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                else:
                    return str(obj)

            doc_size = len(json.dumps(data_to_save, default=json_serializer))
            print(f"🔥 Document size: {doc_size / 1024:.2f} KB")

            if doc_size > 1000000:  # 1MB limit
                print("⚠️ Warning: Document size exceeds 1MB, this may fail")
        except Exception as e:
            print(f"⚠️ Could not calculate document size: {e}")

        # Save to Firestore with explicit float preservation
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
                try:
                    print(f"🔄 Attempting to load TensorFlow model: {model_path}")
                    model = tf.keras.models.load_model(model_path)
                    print(f"✅ Successfully loaded model: {type(model)}")
                    return model
                except Exception as load_error:
                    print(f"❌ Failed to load model {model_path}: {load_error}")
                    print(f"❌ Load error type: {type(load_error).__name__}")
                    # Continue to try other models
            else:
                print(f"⚠️ Modèle {model_name} non trouvé à {model_path}")
                # Continuer pour essayer les modèles par défaut
        
        # Essayer de charger les modèles par défaut (only the models actually uploaded)
        default_models = [
            "Efficient_10unfrozelayers.keras",
            "mobileNet.h5"
        ]
        
        for model_file in default_models:
            model_path = os.path.join(model_dir, model_file)
            if os.path.exists(model_path):
                print(f"✅ Modèle par défaut trouvé: {model_path}")

                try:
                    # Charger selon le type de fichier
                    if model_file.endswith('.h5') or model_file.endswith('.keras'):
                        print(f"🔄 Loading TensorFlow model: {model_path}")
                        model = tf.keras.models.load_model(model_path)
                        print(f"✅ Successfully loaded default model: {type(model)}")
                        return model
                    elif model_file.endswith('.joblib'):
                        print(f"🔄 Loading joblib model: {model_path}")
                        model = joblib.load(model_path)
                        print(f"✅ Successfully loaded joblib model: {type(model)}")
                        return model
                except Exception as load_error:
                    print(f"❌ Failed to load default model {model_path}: {load_error}")
                    print(f"❌ Load error type: {type(load_error).__name__}")
                    # Continue to try next model
        
        # Aucun modèle trouvé, afficher les fichiers disponibles pour le débogage
        print(f"⚠️ Aucun modèle trouvé dans {model_dir}")
        if os.path.exists(model_dir):
            files = os.listdir(model_dir)
            print(f"📁 Fichiers disponibles dans le répertoire: {files}")
        else:
            print(f"📁 Le répertoire {model_dir} n'existe pas")
        
        # Don't fall back to dummy model - raise an error instead
        error_msg = f"❌ No valid models found in {model_dir}"
        if model_name:
            error_msg = f"❌ Failed to load specific model: {model_name}"

        print(f"🔍 Debug: Attempted to load model: {model_name if model_name else 'default models'}")
        print(f"🔍 Debug: Model directory exists: {os.path.exists(model_dir)}")
        if os.path.exists(model_dir):
            available_files = os.listdir(model_dir)
            print(f"🔍 Debug: Available files: {available_files}")

        raise FileNotFoundError(error_msg)
    
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        import traceback
        print(f"❌ Full traceback: {traceback.format_exc()}")
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
                # Ensure proper float conversion with full precision
                prob_value = float(predictions[0][i])
                class_predictions[class_name] = round(prob_value, 6)  # Keep 6 decimal places

            return {
                "prediction": str(class_names[predicted_class]),
                "confidence": round(float(confidence), 6),  # Ensure confidence is also properly formatted
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
                    # Ensure proper float conversion with full precision
                    prob_value = float(probas[i])
                    class_predictions[class_name] = round(prob_value, 6)  # Keep 6 decimal places

                return {
                    "prediction": str(class_names[predicted_class]),
                    "confidence": round(float(confidence), 6),  # Ensure confidence is also properly formatted
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

            # Convert processed image to base64 for Firestore storage
            image_data = convert_image_to_base64(processed_image, "single")

            # Save to Firebase
            firebase_success = save_to_firebase(user_id, prediction_results, image_data)
            prediction_results['saved_to_firebase'] = firebase_success
            if image_data:
                prediction_results['image_stored_in_firestore'] = True
                prediction_results['image_id'] = image_data['image_id']
                prediction_results['image_size_kb'] = round(image_data['size_bytes'] / 1024, 2)
        
        # Return results
        return jsonify(prediction_results)
    
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@prediction_bp.route('/predict-mobilenet', methods=['POST'])
def predict_mobilenet():
    """
    Endpoint pour faire une prédiction avec le modèle MobileNet.

    Accepte deux images via une requête POST multipart/form-data.
    Traite les deux images, fait une prédiction pour chacune, calcule la moyenne
    et retourne un résultat unique.

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

            # Prétraiter l'image pour MobileNet (redimensionner à la taille attendue)
            target_size = (224, 224)  # Taille standard pour MobileNet
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

            # Charger le modèle MobileNet si ce n'est pas déjà fait (direct loading from repository)
            model_name = "mobileNet.h5"  # Use the actual filename in models directory
            if not hasattr(current_app, 'mobilenet_model'):
                print("🔄 Loading MobileNet model from repository...")

                # Load directly from repository (no downloads needed)
                model_path = os.path.join(os.path.dirname(__file__), '..', 'models', model_name)
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found in repository: {model_path}")

                print(f"📁 Loading model from: {model_path}")
                current_app.mobilenet_model = load_model(model_name)
                print(f"✅ MobileNet model loaded: {type(current_app.mobilenet_model)}")

            # Check if we got a dummy model (string) instead of a real model
            if isinstance(current_app.mobilenet_model, str) and current_app.mobilenet_model == "dummy_model":
                # Return realistic dummy predictions for development/testing
                # Create a dummy array with 8 classes (matching your class_names)
                dummy_predictions = np.random.dirichlet(np.ones(8) * 2)  # More realistic distribution
                # Ensure one class has higher probability
                max_idx = np.random.randint(0, 8)
                dummy_predictions[max_idx] = max(dummy_predictions[max_idx], 0.4)
                # Normalize to sum to 1
                dummy_predictions = dummy_predictions / dummy_predictions.sum()
                return dummy_predictions, processed_image_for_storage

            if current_app.mobilenet_model is None:
                raise ValueError(f'Erreur lors du chargement du modèle {model_name}')

            # Faire la prédiction
            predictions = current_app.mobilenet_model.predict(img_array)

            return predictions[0], processed_image_for_storage

        # Traiter les deux images et obtenir les prédictions
        file1.seek(0)  # Remettre le pointeur de fichier au début
        file2.seek(0)
        pred1, processed_img1 = process_and_predict(file1, "left")
        pred2, processed_img2 = process_and_predict(file2, "right")

        # Calculer la moyenne des prédictions
        avg_predictions = (pred1 + pred2) / 2

        # Obtenir les noms de classes
        class_names = current_app.class_names

        # Trouver la classe avec la plus haute probabilité moyenne
        predicted_class_index = np.argmax(avg_predictions)
        predicted_class = class_names[predicted_class_index]
        confidence = round(float(avg_predictions[predicted_class_index]), 6)

        # Créer un dictionnaire avec les classes et leurs probabilités moyennes
        class_predictions = {}
        for i, class_name in enumerate(class_names):
            # Ensure proper float conversion with full precision
            prob_value = float(avg_predictions[i])
            class_predictions[class_name] = round(prob_value, 6)  # Keep 6 decimal places

        # Prepare prediction results
        prediction_results = {
            "prediction": predicted_class,
            "confidence": confidence,
            "class_predictions": class_predictions,
            "is_dummy_prediction": isinstance(current_app.mobilenet_model, str),
            "model_type": "mobilenet_dual_image"
        }

        # Save to Firebase if user_id is provided
        if user_id:
            # Add user_id to prediction results
            prediction_results['user_id'] = user_id

            # Convert processed images to base64 for Firestore storage
            image_data = {}
            left_image_data = convert_image_to_base64(processed_img1, "left")
            right_image_data = convert_image_to_base64(processed_img2, "right")

            if left_image_data:
                image_data['left'] = left_image_data
            if right_image_data:
                image_data['right'] = right_image_data

            # Save to Firebase
            firebase_success = save_to_firebase(user_id, prediction_results, image_data)
            prediction_results['saved_to_firebase'] = firebase_success
            if image_data:
                prediction_results['images_stored_in_firestore'] = True
                prediction_results['image_ids'] = {
                    'left': left_image_data['image_id'] if left_image_data else None,
                    'right': right_image_data['image_id'] if right_image_data else None
                }
                total_size_kb = 0
                if left_image_data:
                    total_size_kb += left_image_data['size_bytes']
                if right_image_data:
                    total_size_kb += right_image_data['size_bytes']
                prediction_results['total_images_size_kb'] = round(total_size_kb / 1024, 2)

        # Retourner les résultats
        return jsonify(prediction_results)

    except Exception as e:
        return jsonify({'error': f'Erreur inattendue: {str(e)}'}), 500


@prediction_bp.route('/predict-efficient', methods=['POST'])
def predict_efficient():
    """
    Endpoint pour faire une prédiction avec le modèle EfficientNet.

    Accepte deux images via une requête POST multipart/form-data.
    Traite les deux images, fait une prédiction pour chacune, calcule la moyenne
    et retourne un résultat unique.

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

            # Charger le modèle EfficientNet si ce n'est pas déjà fait (direct loading from repository)
            model_name = "Efficient_10unfrozelayers.keras"
            if not hasattr(current_app, 'efficient_model'):
                print("🔄 Loading EfficientNet model from repository...")

                # Load directly from repository (no downloads needed)
                model_path = os.path.join(os.path.dirname(__file__), '..', 'models', model_name)
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found in repository: {model_path}")

                print(f"📁 Loading model from: {model_path}")
                current_app.efficient_model = load_model(model_name)
                print(f"✅ EfficientNet model loaded: {type(current_app.efficient_model)}")

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
        confidence = round(float(avg_predictions[predicted_class_index]), 6)

        # Créer un dictionnaire avec les classes et leurs probabilités moyennes
        class_predictions = {}
        for i, class_name in enumerate(class_names):
            # Ensure proper float conversion with full precision
            prob_value = float(avg_predictions[i])
            class_predictions[class_name] = round(prob_value, 6)  # Keep 6 decimal places

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

            # Convert processed images to base64 for Firestore storage
            image_data = {}
            left_image_data = convert_image_to_base64(processed_img1, "left")
            right_image_data = convert_image_to_base64(processed_img2, "right")

            if left_image_data:
                image_data['left'] = left_image_data
            if right_image_data:
                image_data['right'] = right_image_data

            # Save to Firebase
            firebase_success = save_to_firebase(user_id, prediction_results, image_data)
            prediction_results['saved_to_firebase'] = firebase_success
            if image_data:
                prediction_results['images_stored_in_firestore'] = True
                prediction_results['image_ids'] = {
                    'left': left_image_data['image_id'] if left_image_data else None,
                    'right': right_image_data['image_id'] if right_image_data else None
                }
                total_size_kb = 0
                if left_image_data:
                    total_size_kb += left_image_data['size_bytes']
                if right_image_data:
                    total_size_kb += right_image_data['size_bytes']
                prediction_results['total_images_size_kb'] = round(total_size_kb / 1024, 2)

        # Retourner les résultats
        return jsonify(prediction_results)
    
    except Exception as e:
        return jsonify({'error': f'Erreur inattendue: {str(e)}'}), 500

@prediction_bp.route('/check-env', methods=['GET'])
def check_environment():
    """
    Check environment variables for Firebase configuration
    """
    env_vars = {
        'FIREBASE_PROJECT_ID': bool(os.environ.get('FIREBASE_PROJECT_ID')),
        'FIREBASE_PRIVATE_KEY': bool(os.environ.get('FIREBASE_PRIVATE_KEY')),
        'FIREBASE_CLIENT_EMAIL': bool(os.environ.get('FIREBASE_CLIENT_EMAIL')),
        'FIREBASE_PRIVATE_KEY_ID': bool(os.environ.get('FIREBASE_PRIVATE_KEY_ID')),
        'FIREBASE_CLIENT_ID': bool(os.environ.get('FIREBASE_CLIENT_ID')),
    }

    # Show partial values for debugging (without exposing secrets)
    partial_values = {}
    for key in env_vars.keys():
        value = os.environ.get(key, '')
        if value:
            if 'KEY' in key:
                partial_values[key] = f"{value[:20]}...{value[-10:]}" if len(value) > 30 else "***"
            else:
                partial_values[key] = value
        else:
            partial_values[key] = None

    required_vars = ['FIREBASE_PROJECT_ID', 'FIREBASE_PRIVATE_KEY', 'FIREBASE_CLIENT_EMAIL']
    all_required_set = all(env_vars[var] for var in required_vars)

    return jsonify({
        'environment_variables': env_vars,
        'partial_values': partial_values,
        'all_required_set': all_required_set,
        'required_variables': required_vars,
        'firebase_initialized': bool(firebase_admin._apps)
    })

@prediction_bp.route('/get-image/<user_id>/<image_id>', methods=['GET'])
def get_image_from_firestore(user_id, image_id):
    """
    Retrieve a specific image from Firestore by user_id and image_id
    Returns the image as base64 or as a downloadable file
    """
    try:
        # Check if Firebase is initialized
        if not firebase_admin._apps:
            return jsonify({'error': 'Firebase not initialized'}), 500

        db = firestore.client()

        # Query for documents containing the specific image_id
        predictions = db.collection('iris_predictions').where('user_id', '==', user_id).stream()

        for prediction in predictions:
            data = prediction.to_dict()
            processed_images = data.get('processed_images', {})

            # Check if it's a single image
            if isinstance(processed_images, dict) and processed_images.get('image_id') == image_id:
                return jsonify({
                    'image_data': processed_images['image_data'],
                    'image_metadata': {
                        'image_id': processed_images['image_id'],
                        'image_type': processed_images['image_type'],
                        'format': processed_images['format'],
                        'dimensions': processed_images['dimensions'],
                        'size_bytes': processed_images['size_bytes']
                    }
                })

            # Check if it's multiple images (left/right)
            for img_type, img_data in processed_images.items():
                if isinstance(img_data, dict) and img_data.get('image_id') == image_id:
                    return jsonify({
                        'image_data': img_data['image_data'],
                        'image_metadata': {
                            'image_id': img_data['image_id'],
                            'image_type': img_data['image_type'],
                            'format': img_data['format'],
                            'dimensions': img_data['dimensions'],
                            'size_bytes': img_data['size_bytes']
                        }
                    })

        return jsonify({'error': 'Image not found'}), 404

    except Exception as e:
        return jsonify({'error': f'Failed to retrieve image: {str(e)}'}), 500


@prediction_bp.route('/list-predictions/<user_id>', methods=['GET'])
def list_user_predictions(user_id):
    """
    List all predictions for a specific user with image metadata
    """
    try:
        # Check if Firebase is initialized
        if not firebase_admin._apps:
            return jsonify({'error': 'Firebase not initialized'}), 500

        db = firestore.client()

        # Get all predictions for the user
        predictions = db.collection('iris_predictions').where('user_id', '==', user_id).stream()

        result = []
        for prediction in predictions:
            data = prediction.to_dict()

            # Extract basic prediction info
            pred_info = {
                'document_id': prediction.id,
                'timestamp': data.get('timestamp'),
                'prediction': data.get('prediction'),
                'confidence': data.get('confidence'),
                'user_id': data.get('user_id'),
                'images': []
            }

            # Extract image metadata (without the actual image data)
            processed_images = data.get('processed_images', {})
            if isinstance(processed_images, dict):
                if 'image_id' in processed_images:
                    # Single image
                    pred_info['images'].append({
                        'image_id': processed_images['image_id'],
                        'image_type': processed_images['image_type'],
                        'format': processed_images['format'],
                        'dimensions': processed_images['dimensions'],
                        'size_kb': round(processed_images['size_bytes'] / 1024, 2)
                    })
                else:
                    # Multiple images (left/right)
                    for img_type, img_data in processed_images.items():
                        if isinstance(img_data, dict) and 'image_id' in img_data:
                            pred_info['images'].append({
                                'image_id': img_data['image_id'],
                                'image_type': img_data['image_type'],
                                'format': img_data['format'],
                                'dimensions': img_data['dimensions'],
                                'size_kb': round(img_data['size_bytes'] / 1024, 2)
                            })

            result.append(pred_info)

        return jsonify({
            'user_id': user_id,
            'total_predictions': len(result),
            'predictions': result
        })

    except Exception as e:
        return jsonify({'error': f'Failed to list predictions: {str(e)}'}), 500


@prediction_bp.route('/debug-models', methods=['GET'])
def debug_models():
    """
    Debug endpoint to check model loading status and file availability
    """
    try:
        import os
        debug_info = {
            'timestamp': datetime.now().isoformat(),
            'environment': os.environ.get('ENVIRONMENT', 'unknown'),
            'models_directory': {},
            'model_loading_attempts': {},
            'current_models': {}
        }

        # Check models directory
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        model_dir = os.path.abspath(model_dir)

        debug_info['models_directory']['path'] = model_dir
        debug_info['models_directory']['exists'] = os.path.exists(model_dir)

        if os.path.exists(model_dir):
            files = os.listdir(model_dir)
            debug_info['models_directory']['files'] = files

            # Check specific model files
            for filename in ['Efficient_10unfrozelayers.keras', 'mobileNet.h5']:
                file_path = os.path.join(model_dir, filename)
                debug_info['models_directory'][filename] = {
                    'exists': os.path.exists(file_path),
                    'size_mb': round(os.path.getsize(file_path) / (1024*1024), 2) if os.path.exists(file_path) else 0
                }

        # Check current loaded models
        debug_info['current_models']['main_model'] = hasattr(current_app, 'model')
        debug_info['current_models']['efficient_model'] = hasattr(current_app, 'efficient_model')
        debug_info['current_models']['mobilenet_model'] = hasattr(current_app, 'mobilenet_model')

        if hasattr(current_app, 'mobilenet_model'):
            debug_info['current_models']['mobilenet_type'] = type(current_app.mobilenet_model).__name__
            debug_info['current_models']['mobilenet_is_dummy'] = current_app.mobilenet_model == "dummy_model"

        # Test loading MobileNet model
        debug_info['model_loading_attempts']['mobilenet'] = {}
        try:
            test_model = load_model('mobileNet.h5')
            debug_info['model_loading_attempts']['mobilenet']['success'] = test_model != "dummy_model"
            debug_info['model_loading_attempts']['mobilenet']['type'] = type(test_model).__name__
            debug_info['model_loading_attempts']['mobilenet']['is_dummy'] = test_model == "dummy_model"
        except Exception as e:
            debug_info['model_loading_attempts']['mobilenet']['error'] = str(e)

        # Test loading EfficientNet model
        debug_info['model_loading_attempts']['efficient'] = {}
        try:
            test_model = load_model('Efficient_10unfrozelayers.keras')
            debug_info['model_loading_attempts']['efficient']['success'] = test_model != "dummy_model"
            debug_info['model_loading_attempts']['efficient']['type'] = type(test_model).__name__
            debug_info['model_loading_attempts']['efficient']['is_dummy'] = test_model == "dummy_model"
        except Exception as e:
            debug_info['model_loading_attempts']['efficient']['error'] = str(e)

        return jsonify(debug_info)

    except Exception as e:
        return jsonify({
            'error': f'Debug endpoint failed: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500


@prediction_bp.route('/force-download', methods=['POST'])
def force_download():
    """Force download models manually for debugging"""
    try:
        from models.download_models import force_redownload_all_models

        print("🔄 Manual FORCE download triggered via API endpoint")
        result = {
            'timestamp': datetime.now().isoformat(),
            'status': 'starting',
            'message': 'Force redownloading all models (removing corrupted files)...'
        }

        # Force redownload all models
        download_results = force_redownload_all_models()

        result['status'] = 'completed'
        result['message'] = 'Force download process completed.'
        result['download_results'] = download_results

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Force download failed: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500


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
                'message': 'Check environment variables or credentials file',
                'help': 'Visit /api/check-env to see environment variable status'
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

        # Test Storage connection (still available for other uses)
        try:
            bucket = storage.bucket()
            storage_status = f'connected to bucket: {bucket.name}'
        except Exception as e:
            storage_status = f'error: {str(e)}'

        return jsonify({
            'firebase_initialized': True,
            'firestore_status': firestore_status,
            'storage_status': storage_status,
            'storage_note': 'Images now stored in Firestore, not Storage',
            'app_name': firebase_admin._apps['[DEFAULT]'].name if '[DEFAULT]' in firebase_admin._apps else 'unknown'
        })

    except Exception as e:
        return jsonify({
            'error': f'Firebase test failed: {str(e)}',
            'firebase_initialized': bool(firebase_admin._apps)
        }), 500









