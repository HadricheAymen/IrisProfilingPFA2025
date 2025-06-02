from app import app
from waitress import serve
import os
import logging

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)

if __name__ == "__main__":
    # Railway automatically sets PORT environment variable
    port = int(os.environ.get("PORT", 8080))
    host = "0.0.0.0"  # Railway requires 0.0.0.0
    flask_env = os.environ.get("FLASK_ENV", "production")

    print(f"🚀 Starting server on {host}:{port}")
    print(f" Environment: {flask_env}")
    print(f"🔥 Firebase credentials: {os.environ.get('FIREBASE_CREDENTIALS', 'profilingiris-firebase-adminsdk-fbsvc-5a10da7e73.json')}")

    if flask_env == "development":
        app.run(debug=True, host=host, port=port)
    else:
        # Production server optimized for Railway
        serve(
            app,
            host=host,
            port=port,
            threads=4,  # Optimized for Railway's resources
            connection_limit=500,  # Conservative limit for Railway
            cleanup_interval=30,
            channel_timeout=120
        )