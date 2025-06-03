# Railway Deployment Guide

## 🚀 Deploy Your Iris Profiling Backend to Railway

### Prerequisites

1. **GitHub Account**: Your code should be in a GitHub repository
2. **Railway Account**: Sign up at [railway.app](https://railway.app)
3. **Firebase Project**: With service account credentials

### Step-by-Step Deployment

#### 1. Prepare Your Repository

Make sure your repository has these files (already created):
- ✅ `railway.json` - Railway configuration
- ✅ `nixpacks.toml` - Build configuration
- ✅ `requirements.txt` - Python dependencies
- ✅ `run.py` - Application entry point
- ✅ `app.py` - Flask application
- ✅ `profilingiris-firebase-adminsdk-fbsvc-5a10da7e73.json` - Firebase credentials

#### 2. Push to GitHub

```bash
# Add all files
git add .

# Commit changes
git commit -m "Prepare for Railway deployment"

# Push to GitHub
git push origin main
```

#### 3. Deploy on Railway

1. Go to [Railway](https://railway.app)
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Choose your repository
5. Railway will automatically:
   - Detect it's a Python Flask app
   - Install dependencies from `requirements.txt`
   - Use the configuration from `railway.json`
   - Start your app with `python run.py`

#### 4. Configure Environment Variables

In your Railway project dashboard, go to **Variables** and add:

```
FLASK_ENV=production
API_VERSION=1.0.0
ALLOWED_ORIGINS=*
FIREBASE_TYPE=service_account
FIREBASE_PROJECT_ID=profilingiris
FIREBASE_PRIVATE_KEY_ID=5a10da7e73b1009245acce86e3378aefa77978e8
FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC+Z8...truncated...vOzEkUw==\n-----END PRIVATE KEY-----\n"
FIREBASE_CLIENT_EMAIL=firebase-adminsdk-fbsvc@profilingiris.iam.gserviceaccount.com
FIREBASE_CLIENT_ID=112585792494326703507
FIREBASE_AUTH_URI=https://accounts.google.com/o/oauth2/auth
FIREBASE_TOKEN_URI=https://oauth2.googleapis.com/token
FIREBASE_AUTH_PROVIDER_X509_CERT_URL=https://www.googleapis.com/oauth2/v1/certs
FIREBASE_CLIENT_X509_CERT_URL=https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-fbsvc%40profilingiris.iam.gserviceaccount.com
FIREBASE_UNIVERSE_DOMAIN=googleapis.com
```

**For production, replace `ALLOWED_ORIGINS=*` with your actual frontend domain:**
```
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
```

#### 5. Monitor Deployment

1. Watch the **Deployments** tab for build progress
2. Check **Logs** for any errors
3. Once deployed, you'll get a URL like: `https://your-app-name.railway.app`

#### 6. Test Your Deployment

Visit your Railway URL:
- `https://your-app-name.railway.app/` - API info
- `https://your-app-name.railway.app/health` - Health check

### 🔧 Configuration Files Explained

#### `railway.json`
```json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "python run.py",
    "healthcheckPath": "/health"
  }
}
```

#### `nixpacks.toml`
Configures the build environment with required system packages:
- Python 3.11
- OpenCV
- dlib
- Other ML dependencies

### 🔍 Troubleshooting

#### Common Issues:

1. **Build Fails**:
   - Check `requirements.txt` for version conflicts
   - Review build logs in Railway dashboard

2. **App Won't Start**:
   - Ensure `run.py` is in the root directory
   - Check that `PORT` environment variable is used correctly

3. **Firebase Errors**:
   - Verify Firebase credentials file is in repository
   - Check `FIREBASE_CREDENTIALS` environment variable

4. **Model Loading Issues**:
   - Ensure model files are in the `models/` directory
   - Check file paths in your code

#### Viewing Logs:
```bash
# In Railway dashboard, go to your project
# Click on "Logs" to see real-time application logs
```

### 🔒 Security Considerations

1. **Environment Variables**: Never commit sensitive data to GitHub
2. **CORS**: Set specific origins in production
3. **Firebase**: Use service account with minimal permissions
4. **Rate Limiting**: Consider adding rate limiting for production

### 📊 Monitoring

Railway provides:
- **Metrics**: CPU, Memory, Network usage
- **Logs**: Real-time application logs
- **Health Checks**: Automatic monitoring via `/health` endpoint

### 🔄 Updates

To update your deployment:
```bash
git add .
git commit -m "Update application"
git push origin main
```

Railway will automatically redeploy when you push to GitHub.

### 💡 Tips

1. **Custom Domain**: You can add a custom domain in Railway settings
2. **Environment Branches**: Use different branches for staging/production
3. **Database**: Railway offers PostgreSQL if you need a database
4. **Scaling**: Railway automatically handles scaling based on traffic

### 🆘 Support

- Railway Documentation: [docs.railway.app](https://docs.railway.app)
- Railway Discord: [discord.gg/railway](https://discord.gg/railway)
- GitHub Issues: For application-specific problems

