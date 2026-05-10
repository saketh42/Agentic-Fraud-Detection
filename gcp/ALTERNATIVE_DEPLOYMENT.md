# Alternative Deployment Options

Since Docker build requires Windows Docker Desktop integration, here are alternative free deployment options:

## Option 1: Heroku (Recommended - Easiest)

### Step 1: Install Heroku CLI
```bash
# Install Heroku CLI (Linux/WSL)
curl https://cli-assets.heroku.com/install.sh | sh

# Or on Windows, download from https://devcenter.heroku.com/articles/heroku-cli
```

### Step 2: Deploy API Server
```bash
cd /mnt/c/Users/Priti/Desktop/Agentic-Fraud-Detection

# Create Heroku app
heroku create fraud-detection-api

# Add requirements file for Heroku
cat > requirements_heroku.txt << EOF
flask>=3.0.0
gunicorn>=21.0.0
requests>=2.31.0
pydantic>=2.0.0
python-multipart>=0.0.9
EOF

# Deploy
git push heroku main
```

### Step 3: Deploy Streamlit Frontend
```bash
# Create another Heroku app for frontend
heroku create fraud-detection-ui

# Set API URL as environment variable
heroku config:set API_BASE_URL=https://fraud-detection-api.herokuapp.com

# Deploy
git push heroku main
```

---

## Option 2: Railway.app (Modern, Easy)

1. Go to https://railway.app
2. Connect your GitHub repository
3. Deploy automatically from GitHub

---

## Option 3: Render.com (Free Tier Available)

1. Go to https://render.com
2. Create Web Service for API
3. Set start command: `gunicorn simple_api:app`
4. Deploy!

---

## Option 4: PythonAnywhere (Simple Python Hosting)

1. Create free account at https://pythonanywhere.com
2. Upload your code via Files tab
3. Set up a Flask web app in the Web tab
4. Use their free web hosting!

---

## Option 5: Google Cloud Run via Cloud Shell (No Local Docker)

### Use Cloud Shell in Browser

1. Go to https://shell.cloud.google.com
2. Clone your GitHub repo or upload files
3. Run deployment commands directly in browser

---

## Quick Deploy Instructions (Heroku)

### For API Server:

```bash
# Create app
heroku create fraud-detection-api-demo --buildpack heroku/python

# Configure
heroku config:set PORT=8000

# Create Procfile for API
echo "web: gunicorn simple_api:app --bind 0.0.0.0:$PORT" > Procfile.api

# Deploy
git push heroku main
```

### For Streamlit Frontend:

```bash
# Create app
heroku create fraud-detection-ui-demo --buildpack heroku/python

# Create Procfile for Streamlit
echo "web: streamlit run app/frontend/streamlit_app.py --server.port \$PORT --server.address 0.0.0.0" > Procfile.ui

# Add buildpack for Python with requirements
heroku buildpacks:add heroku/python

# Set API URL
heroku config:set API_BASE_URL=https://fraud-detection-api-demo.herokuapp.com

# Deploy
git push heroku main
```

---

## Summary

| Platform | Free Tier | Difficulty | URL |
|----------|----------|------------|-----|
| Heroku | 550 hours/month | Easy | heroku.com |
| Railway | $5/month credit | Easy | railway.app |
| Render | 750 hours/month | Easy | render.com |
| PythonAnywhere | Free | Easy | pythonanywhere.com |
| GCP Cloud Shell | Limited | Medium | cloud.google.com/shell |

---

## Recommended Approach

1. **For quickest setup**: Use Heroku
2. **For GCP integration**: Use GCP Cloud Shell
3. **For simplicity**: Use PythonAnywhere

Which option would you like to proceed with?