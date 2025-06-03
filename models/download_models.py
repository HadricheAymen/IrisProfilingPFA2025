"""
Download large model files at runtime to avoid GitHub file size limits
"""
import os
import requests
from pathlib import Path
import logging
import bz2
import shutil
import zipfile

logger = logging.getLogger(__name__)

# Model download URLs
MODEL_URLS = {
    'shape_predictor_68_face_landmarks.dat': 'https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2',
    # EfficientNet model from GitHub Release (direct file)
    'Efficient_10unfrozelayers.keras': 'https://github.com/HadricheAymen/IrisProfilingPFA2025/releases/download/v1.0.0/Efficient_10unfrozelayers.keras'
}

# ZIP archives containing ML models (currently none - using direct downloads)
ZIP_MODELS = {
    # No ZIP files needed - using direct downloads
}

# Direct file downloads (for files hosted externally)
DIRECT_DOWNLOADS = {
    # MobileNet model from Google Drive
    'mobileNet.h5': 'https://drive.google.com/uc?export=download&id=1mayaqjO2xqegcwwAal8V8a8NAO4fDnx4'
}

# Fallback: If models can't be downloaded, use dummy models
FALLBACK_ENABLED = True

def download_file(url, filepath):
    """Download a file from URL to filepath"""
    try:
        logger.info(f"Downloading {filepath} from {url}")

        # Special handling for Google Drive URLs
        if 'drive.google.com' in url:
            return download_google_drive_file(url, filepath)

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


def download_google_drive_file(url, filepath):
    """Download a file from Google Drive with virus scan handling"""
    try:
        logger.info(f"Downloading from Google Drive: {filepath}")

        session = requests.Session()
        response = session.get(url, stream=True)

        # Check if we got a virus scan warning page
        if 'virus scan warning' in response.text.lower() or 'download anyway' in response.text.lower():
            logger.info("Handling Google Drive virus scan warning...")

            # Look for the download confirmation link
            import re
            confirm_pattern = r'href="(/uc\?export=download[^"]*)"'
            match = re.search(confirm_pattern, response.text)

            if match:
                confirm_url = 'https://drive.google.com' + match.group(1).replace('&amp;', '&')
                logger.info(f"Found confirmation URL: {confirm_url}")
                response = session.get(confirm_url, stream=True)
            else:
                logger.warning("Could not find download confirmation link")
                return False

        response.raise_for_status()

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Successfully downloaded from Google Drive: {filepath}")
        return True

    except Exception as e:
        logger.error(f"Failed to download from Google Drive {filepath}: {e}")
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


def extract_zip(zip_path, extract_to_dir):
    """Extract a ZIP file to specified directory"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_dir)
        return True
    except Exception as e:
        logger.error(f"Failed to extract {zip_path}: {e}")
        return False

def ensure_models_downloaded():
    """Ensure all required models are downloaded"""
    models_dir = Path(__file__).parent

    # Download individual model files
    for filename, url in MODEL_URLS.items():
        filepath = models_dir / filename

        if not filepath.exists() or filepath.stat().st_size == 0:  # Also check for empty files
            logger.info(f"Model {filename} not found or empty, downloading...")

            if url.endswith('.bz2'):
                # Handle compressed files (like dlib models)
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
                # Handle regular files (like .h5, .keras models)
                success = download_file(url, filepath)
                if success:
                    logger.info(f"Successfully downloaded {filename}")
                else:
                    logger.warning(f"Failed to download {filename}, app may not work properly")
        else:
            logger.info(f"Model {filename} already exists")

    # Download and extract ZIP archives containing multiple models
    for zip_name, zip_info in ZIP_MODELS.items():
        zip_path = models_dir / zip_name
        url = zip_info['url']
        contained_files = zip_info['contains']

        # Check if any of the contained files are missing or empty
        missing_files = []
        for contained_file in contained_files:
            file_path = models_dir / contained_file
            if not file_path.exists() or file_path.stat().st_size == 0:
                missing_files.append(contained_file)

        if missing_files:
            logger.info(f"Missing model files {missing_files}, downloading ZIP archive {zip_name}")

            # Download ZIP file
            success = download_file(url, zip_path)

            if success:
                # Extract ZIP file
                logger.info(f"Extracting {zip_path}")
                extract_success = extract_zip(zip_path, models_dir)

                if extract_success:
                    # Remove ZIP file after extraction
                    os.remove(zip_path)
                    logger.info(f"Successfully extracted models from {zip_name}")

                    # Verify extracted files
                    for contained_file in contained_files:
                        file_path = models_dir / contained_file
                        if file_path.exists() and file_path.stat().st_size > 0:
                            logger.info(f"✅ {contained_file} extracted successfully")
                        else:
                            logger.warning(f"❌ {contained_file} extraction failed")
                else:
                    logger.warning(f"Failed to extract {zip_name}")
            else:
                logger.warning(f"Failed to download {zip_name}, ML models may not work properly")
        else:
            logger.info(f"All files from {zip_name} already exist")

    # Download direct files (like Google Drive links)
    for filename, url in DIRECT_DOWNLOADS.items():
        filepath = models_dir / filename

        if not filepath.exists() or filepath.stat().st_size == 0:
            logger.info(f"Direct download: {filename} not found or empty, downloading...")
            success = download_file(url, filepath)
            if success:
                logger.info(f"✅ Successfully downloaded {filename} directly")
            else:
                logger.warning(f"❌ Failed to download {filename} directly")
        else:
            logger.info(f"Direct download: {filename} already exists")

if __name__ == "__main__":
    ensure_models_downloaded()
