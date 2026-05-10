# GCP Deployment Guide - Agentic Fraud Detection Demo

## Prerequisites

Before deploying to GCP, ensure you have:

1. **Google Cloud Platform Account**
   - Sign up at https://console.cloud.google.com
   - Create a new project or use existing one

2. **Google Cloud SDK (gcloud CLI)**
   - Install from: https://cloud.google.com/sdk/docs/install
   - After installation, authenticate:
     ```bash
     gcloud auth login
     gcloud init
     ```

3. **Docker** (for local testing)
   - Install Docker Desktop from https://docker.com

---

## Step 1: Project Setup

### 1.1 Set your project ID
```bash
# Replace 'your-project-id' with your actual GCP project ID
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID
```

### 1.2 Enable required APIs
```bash
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable cloudsql.googleapis.com
```

---

## Step 2: Local Testing

Test your application locally using Docker before deploying to GCP:

### 2.1 Build Docker images locally
```bash
cd /mnt/c/Users/Priti/Desktop/Agentic-Fraud-Detection

# Build API image
docker build -f gcp/Dockerfile.api -t fraud-detection-api:latest .

# Build Frontend image
docker build -f gcp/Dockerfile.frontend -t fraud-detection-ui:latest .
```

### 2.2 Run locally
```bash
# Run API server
docker run -p 8000:8000 fraud-detection-api:latest

# In another terminal, run frontend
docker run -p 8501:8501 fraud-detection-ui:latest
```

---

## Step 3: Deploy to GCP

### 3.1 Create Artifact Registry repository
```bash
gcloud artifacts repositories create fraud-detection-repo \
    --repository-format=docker \
    --location=us-central1
```

### 3.2 Configure Docker authentication
```bash
gcloud auth configure-docker us-central1-docker.pkg.dev
```

### 3.3 Deploy API Server

```bash
# Build and push API image
gcloud builds submit \
    --tag us-central1-docker.pkg.dev/$PROJECT_ID/fraud-detection-repo/fraud-detection-api:v1 \
    --file gcp/Dockerfile.api .

# Deploy to Cloud Run
gcloud run deploy fraud-detection-api \
    --image us-central1-docker.pkg.dev/$PROJECT_ID/fraud-detection-repo/fraud-detection-api:v1 \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 8000 \
    --memory 512Mi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 10
```

### 3.4 Deploy Streamlit Frontend

```bash
# Get the API URL from previous deployment
API_URL=$(gcloud run services describe fraud-detection-api --region us-central1 --format 'value(status.url)')

# Build and push Frontend image
gcloud builds submit \
    --tag us-central1-docker.pkg.dev/$PROJECT_ID/fraud-detection-repo/fraud-detection-ui:v1 \
    --file gcp/Dockerfile.frontend .

# Deploy to Cloud Run with API URL
gcloud run deploy fraud-detection-ui \
    --image us-central1-docker.pkg.dev/$PROJECT_ID/fraud-detection-repo/fraud-detection-ui:v1 \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 8501 \
    --memory 1Gi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 5 \
    --set-env-vars="API_BASE_URL=$API_URL"
```

---

## Step 4: Verify Deployment

### 4.1 Get service URLs
```bash
# API URL
API_URL=$(gcloud run services describe fraud-detection-api --region us-central1 --format 'value(status.url)')
echo "API Server: $API_URL"

# Frontend URL
UI_URL=$(gcloud run services describe fraud-detection-ui --region us-central1 --format 'value(status.url)')
echo "Streamlit UI: $UI_URL"
```

### 4.2 Test endpoints
```bash
# Test API health
curl $API_URL/health

# Test prediction endpoint
curl -X POST $API_URL/api/predict/single \
  -H "Content-Type: application/json" \
  -d '{"fraud_probability": 0.85}'
```

---

## Step 5: Custom Domain (Optional)

To use a custom domain:

1. Map custom domain to Cloud Run:
```bash
gcloud run domain-mappings create --service fraud-detection-api --domain api.yourdomain.com
gcloud run domain-mappings create --service fraud-detection-ui --domain yourdomain.com
```

2. Update DNS records as shown in GCP Console

---

## Monitoring and Logs

### View logs
```bash
# API logs
gcloud run logs read fraud-detection-api --region us-central1

# Frontend logs
gcloud run logs read fraud-detection-ui --region us-central1
```

### Monitor performance
1. Go to Cloud Console: https://console.cloud.google.com/run
2. Select your service
3. View metrics, logs, and performance

---

## Cost Optimization Tips

1. **Set min instances to 0** - Scales to zero when not in use
2. **Use regional endpoints** - Reduces latency
3. **Monitor usage** - Check GCP Console regularly
4. **Use budget alerts** - Set up alerts to avoid unexpected charges

---

## Troubleshooting

### Container fails to start
```bash
# Check logs
gcloud run logs read fraud-detection-api --region us-central1 --limit 50

# Common issues:
# - Port mismatch (check Dockerfile EXPOSE)
# - Missing dependencies (check requirements.txt)
# - Environment variables not set
```

### Permission errors
```bash
# Ensure you have required IAM roles
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="user:your-email@example.com" \
    --role="roles/run.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="user:your-email@example.com" \
    --role="roles/artifactregistry.admin"
```

### Build failures
```bash
# Test build locally first
docker build -f gcp/Dockerfile.api -t test-build:latest .

# Check for syntax errors in Dockerfile
```

---

## Next Steps

1. **Set up CI/CD** - Use Cloud Build for automatic deployments
2. **Add custom domain** - Use Cloud Run with custom domains
3. **Implement authentication** - Add user authentication
4. **Set up monitoring** - Use Cloud Monitoring dashboards
5. **Configure alerts** - Set up budget and performance alerts

---

## Files Created for GCP Deployment

```
gcp/
├── DEPLOYMENT.md        # This file
├── deploy.sh            # Automated deployment script
├── Dockerfile.api       # API server container
├── Dockerfile.frontend # Streamlit frontend container
└── cloudbuild.yaml      # Cloud Build configuration (optional)
```

---

## Need Help?

- GCP Documentation: https://cloud.google.com/run/docs
- Cloud Run Quickstart: https://cloud.google.com/run/docs/quickstarts/prebuilt/container-image
- Community Support: https://stackoverflow.com/questions/tagged/google-cloud-run