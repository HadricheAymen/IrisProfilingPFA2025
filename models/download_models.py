"""
Download large model files at runtime to avoid GitHub file size limits
"""
import os
import requests
from pathlib import Path
import logging
import bz2
import shutil

logger = logging.getLogger(__name__)

# Model download URLs
MODEL_URLS = {
    'shape_predictor_68_face_landmarks.dat': 'https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2',
    # Add your ML models here - you'll need to host them somewhere accessible
    # 'Efficient_10unfrozelayers.keras': 'YOUR_DOWNLOAD_URL_HERE',
    # 'mobileNet.h5': 'YOUR_DOWNLOAD_URL_HERE'
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

def extract_bz2(compressed_path, output_path):
    """Extract a bz2 compressed file"""
    try:
        with open(output_path, 'wb') as new_file, bz2.BZ2File(compressed_path, 'rb') as file:
            for data in iter(lambda: file.read(100 * 1024), b''):
                new_file.write(data)
        return True
    except Exception as e:
        logger.error(f"Failed to extract {compressed_path}: {e}")
        return False

def ensure_models_downloaded():
    """Ensure all required models are downloaded"""
    models_dir = Path(__file__).parent

    for filename, url in MODEL_URLS.items():
        filepath = models_dir / filename

        if not filepath.exists():
            logger.info(f"Model {filename} not found, downloading...")

            # Download compressed file
            compressed_path = filepath.with_suffix(filepath.suffix + '.bz2')
            success = download_file(url, compressed_path)

            if success:
                # Extract the file
                logger.info(f"Extracting {compressed_path}")
                extract_success = extract_bz2(compressed_path, filepath)

                if extract_success:
                    # Remove compressed file
                    os.remove(compressed_path)
                    logger.info(f"Successfully extracted {filename}")
                else:
                    logger.warning(f"Failed to extract {filename}")
            else:
                logger.warning(f"Failed to download {filename}, app may not work properly")
        else:
            logger.info(f"Model {filename} already exists")

if __name__ == "__main__":
    ensure_models_downloaded()
