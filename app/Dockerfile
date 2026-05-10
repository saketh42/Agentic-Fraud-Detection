FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt requirements-web.txt ./
RUN pip install -r requirements.txt
RUN pip install -r requirements-web.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.api.server:app", "--host", "0.0.0.0", "--port", "8000"]