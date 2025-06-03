from flask import Flask, jsonify
from flask_cors import CORS
from waitress import serve
from api.iris_extraction import iris_bp
from api.prediction import prediction_bp
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Download models if needed (for Railway deployment)
try:
    from models.download_models import ensure_models_downloaded
    ensure_models_downloaded()
except Exception as e:
    logger.warning(f"Model download check failed: {e}")

# Configure CORS based on environment
if os.environ.get('FLASK_ENV') == 'development':
    # Development: Allow all origins
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    logger.info("🔧 CORS configured for development (all origins)")
else:
    # Production: Restrict to specific origins
    allowed_origins = os.environ.get('ALLOWED_ORIGINS', '*').split(',')
    CORS(app, resources={r"/api/*": {"origins": allowed_origins}})
    logger.info(f"🔒 CORS configured for production (origins: {allowed_origins})")

# Register blueprints
app.register_blueprint(iris_bp, url_prefix='/api')
app.register_blueprint(prediction_bp, url_prefix='/api')

# Define class names for the EfficientNet model
app.class_names = ['Flower-Jewel', 'Flower-Stream', 'Shaker-Stream', 'Flower', 'Jewel', 'Shaker', 'Shaker-Jewel', 'Stream']

# Enhanced health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    import psutil
    import tensorflow as tf

    try:
        # Check system resources
        memory_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent()

        # Check if models are loaded
        models_status = {
            'main_model': hasattr(app, 'model'),
            'efficient_model': hasattr(app, 'efficient_model'),
            'mobilenet_model': hasattr(app, 'mobilenet_model')
        }

        # Check Firebase connection
        firebase_status = 'unknown'
        try:
            import firebase_admin
            firebase_status = 'connected' if firebase_admin._apps else 'not_initialized'
        except Exception:
            firebase_status = 'error'

        # Check dlib and model availability
        dlib_status = 'unknown'
        model_file_status = 'unknown'
        try:
            import dlib
            dlib_status = 'available'

            # Check if shape predictor model exists
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'shape_predictor_68_face_landmarks.dat')
            model_file_status = 'available' if os.path.exists(model_path) else 'missing'
        except ImportError:
            dlib_status = 'not_installed'

        return jsonify({
            'status': 'healthy',
            'version': os.environ.get('API_VERSION', '1.0.0'),
            'environment': os.environ.get('FLASK_ENV', 'production'),
            'tensorflow_version': tf.__version__,
            'system': {
                'memory_usage_percent': memory_usage,
                'cpu_usage_percent': cpu_usage
            },
            'models': models_status,
            'firebase': firebase_status,
            'dlib': dlib_status,
            'shape_predictor_model': model_file_status,
            'timestamp': os.popen('date').read().strip()
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'version': os.environ.get('API_VERSION', '1.0.0')
        }), 500

# Add a simple root endpoint
@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'message': 'Iris Profiling API',
        'version': os.environ.get('API_VERSION', '1.0.0'),
        'endpoints': {
            'health': '/health',
            'iris_extraction': '/api/extract-iris',
            'prediction': '/api/predict',
            'efficient_prediction': '/api/predict-efficient',
            'mobilenet_prediction': '/api/predict-mobilenet'
        }
    })

if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.environ.get('PORT', 5000))
    
    if debug_mode:
        app.run(debug=True, host='0.0.0.0', port=port)
    else:
        print(f"Starting production server on port {port}")
        serve(app, host='0.0.0.0', port=port)



