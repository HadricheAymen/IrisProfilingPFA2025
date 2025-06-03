#!/bin/bash
# Optimized build script for Railway deployment

echo "🚀 Starting optimized build process..."

# Set environment variables for faster builds
export PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export CMAKE_ARGS="-DDLIB_NO_GUI_SUPPORT=ON"
export DLIB_NO_GUI_SUPPORT=1

echo "📦 Installing build dependencies..."
pip install --upgrade pip wheel setuptools

echo "🔧 Installing dlib (this may take a while)..."
pip install --no-cache-dir --timeout 1000 dlib

echo "🧠 Installing TensorFlow CPU..."
pip install --no-cache-dir --timeout 1000 tensorflow-cpu

echo "👁️ Installing OpenCV..."
pip install --no-cache-dir --timeout 1000 opencv-python-headless

echo "📋 Installing remaining requirements..."
pip install --no-cache-dir -r requirements.txt

echo "✅ Build completed successfully!"
