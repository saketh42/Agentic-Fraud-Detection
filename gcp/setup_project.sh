#!/bin/bash
# Deploy to GCP Project: fraud-495917
# Run these commands in Cloud Shell

PROJECT_ID="fraud-495917"
REGION="us-central1"

echo "=========================================="
echo "Deploying to Project: $PROJECT_ID"
echo "=========================================="

# Step 1: Configure project
echo "Step 1: Setting project..."
gcloud config set project $PROJECT_ID

# Step 2: Enable APIs
echo "Step 2: Enabling APIs..."
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com --quiet

# Step 3: Create Artifact Registry
echo "Step 3: Creating Artifact Registry..."
gcloud artifacts repositories create fraud-detection-repo \
    --repository-format=docker \
    --location=$REGION \
    --description="Fraud detection demo" \
    --quiet || echo "Repo exists, continuing..."

# Step 4: Authenticate Docker
echo "Step 4: Authenticating Docker..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

echo ""
echo "=========================================="
echo "Configuration complete!"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "=========================================="