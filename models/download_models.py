"""
Download large model files at runtime to avoid GitHub file size limits
"""
import os
import requests
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Model download URLs (for models that need to be downloaded)
MODEL_URLS = {
    # Add your Keras model URL here if you host it somewhere
    # 'Efficient_10unfrozelayers.keras': 'YOUR_MODEL_URL_HERE'
    # Note: dlib shape predictor is no longer needed as we use OpenCV for face/eye detection
}

def download_file(url, filepath):
    """Download a file from URL to filepath"""
    try:
        logger.info(f"Downloading {filepath} from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Successfully downloaded {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {filepath}: {e}")
        return False

def ensure_models_downloaded():
    """Ensure all required models are downloaded"""
    models_dir = Path(__file__).parent
    
    for filename, url in MODEL_URLS.items():
        filepath = models_dir / filename
        
        if not filepath.exists():
            logger.info(f"Model {filename} not found, downloading...")
            success = download_file(url, filepath)
            if not success:
                logger.warning(f"Failed to download {filename}, app may not work properly")
        else:
            logger.info(f"Model {filename} already exists")

if __name__ == "__main__":
    ensure_models_downloaded()
