#!/bin/bash
# Simple GCP Deployment - Run Line by Line

echo "=========================================="
echo "GCP CLOUD SHELL DEPLOYMENT"
echo "=========================================="
echo ""

# IMPORTANT: Run these commands one by one in GCP Cloud Shell
# https://shell.cloud.google.com

# STEP 1: Set your project ID
echo "STEP 1: Set your GCP Project ID"
echo "Run: gcloud config set project YOUR_PROJECT_ID"
echo ""

# STEP 2: Enable APIs
echo "STEP 2: Enable required APIs"
echo "Run these commands:"
echo "  gcloud services enable run.googleapis.com"
echo "  gcloud services enable cloudbuild.googleapis.com"
echo "  gcloud services enable artifactregistry.googleapis.com"
echo ""

# STEP 3: Create repository
echo "STEP 3: Create Artifact Registry"
echo "Run: gcloud artifacts repositories create fraud-detection-repo --repository-format=docker --location=us-central1"
echo ""

# STEP 4: Authenticate Docker
echo "STEP 4: Authenticate Docker"
echo "Run: gcloud auth configure-docker us-central1-docker.pkg.dev"
echo ""

# STEP 5: Deploy API
echo "STEP 5: Build and Deploy API Server"
echo "Run these commands:"
echo "  gcloud builds submit --tag us-central1-docker.pkg.dev/\$PROJECT_ID/fraud-detection-repo/fraud-detection-api:v1 --file gcp/Dockerfile.api ."
echo "  gcloud run deploy fraud-detection-api --image us-central1-docker.pkg.dev/\$PROJECT_ID/fraud-detection-repo/fraud-detection-api:v1 --platform managed --region us-central1 --allow-unauthenticated --port 8000"
echo ""

# STEP 6: Get API URL
echo "STEP 6: Get API URL"
echo "Run: gcloud run services describe fraud-detection-api --region us-central1 --format 'value(status.url)'"
echo "  Copy this URL - you'll need it for the frontend"
echo ""

# STEP 7: Deploy Frontend
echo "STEP 7: Build and Deploy Streamlit Frontend"
echo "Replace YOUR_API_URL with the URL from Step 6"
echo "Run these commands:"
echo "  gcloud builds submit --tag us-central1-docker.pkg.dev/\$PROJECT_ID/fraud-detection-repo/fraud-detection-ui:v1 --file gcp/Dockerfile.frontend ."
echo "  gcloud run deploy fraud-detection-ui --image us-central1-docker.pkg.dev/\$PROJECT_ID/fraud-detection-repo/fraud-detection-ui:v1 --platform managed --region us-central1 --allow-unauthenticated --port 8501 --set-env-vars=API_BASE_URL=YOUR_API_URL"
echo ""

# STEP 8: Get Final URLs
echo "STEP 8: Get Your Service URLs"
echo "Run:"
echo "  API: gcloud run services describe fraud-detection-api --region us-central1 --format 'value(status.url)'"
echo "  UI: gcloud run services describe fraud-detection-ui --region us-central1 --format 'value(status.url)'"
echo ""

echo "=========================================="
echo "After deployment, your services will be live at:"
echo "  API: https://fraud-detection-api-XXXXX-uc.a.run.app"
echo "  UI:  https://fraud-detection-ui-XXXXX-uc.a.run.app"
echo "=========================================="
echo ""
echo "Test API: curl YOUR_API_URL/health"
echo "Open browser to: YOUR_UI_URL"
echo ""
echo "Good luck! 🚀"