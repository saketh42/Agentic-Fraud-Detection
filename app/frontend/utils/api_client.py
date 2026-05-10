"""
API client utility for the fraud detection demo
"""
import requests
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class FraudDetectionAPIClient:
    """API client for the fraud detection system"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def get_system_status(self):
        """Get system status"""
        try:
            response = requests.get(f"{self.base_url}/api/status")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to get system status: {str(e)}")
    
    def predict_fraud(self, transaction_data):
        """Predict fraud for a single transaction"""
        try:
            response = requests.post(
                f"{self.base_url}/api/predict/single",
                json=transaction_data
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to predict fraud: {str(e)}")
    
    def predict_batch_fraud(self, transactions):
        """Predict fraud for a batch of transactions"""
        try:
            response = requests.post(
                f"{self.base_url}/api/predict/batch",
                json={"transactions": transactions}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to predict batch fraud: {str(e)}")
    
    def get_metrics(self):
        """Get model metrics"""
        try:
            response = requests.get(f"{self.base_url}/api/metrics")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to get metrics: {str(e)}")
    
    def get_drift_status(self):
        """Get drift detection status"""
        try:
            response = requests.get(f"{self.base_url}/api/drift")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to get drift status: {str(e)}")
    
    def run_pipeline(self):
        """Run the full pipeline"""
        try:
            response = requests.get(f"{self.base_url}/api/pipeline/run")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to run pipeline: {str(e)}")

# Create a global instance
api_client = FraudDetectionAPIClient()