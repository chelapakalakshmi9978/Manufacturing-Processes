"""
Configuration for IBM Granite-powered Manufacturing Quality Control System
"""

import os
from dotenv import load_dotenv

load_dotenv()

# IBM Watson X AI Configuration
IBM_API_KEY = os.getenv("IBM_API_KEY", "bpSKgmuggXW8pMu1rc0G0XL22LglM8dkx2kYkQxoxjtZ")
IBM_PROJECT_ID = os.getenv("IBM_PROJECT_ID", "3cb648e7-2430-4e25-86ab-bc51d756ef68")
IBM_WATSONX_URL = os.getenv("IBM_WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

# IBM Granite Model — granite-4-h-small is the available Granite instruct model
GRANITE_MODEL_ID = "ibm/granite-4-h-small"

# Manufacturing Thresholds
QUALITY_THRESHOLDS = {
    "temperature": {"min": 180, "max": 220, "unit": "°C"},
    "pressure": {"min": 95, "max": 105, "unit": "bar"},
    "vibration": {"min": 0.0, "max": 2.5, "unit": "mm/s"},
    "humidity": {"min": 30, "max": 60, "unit": "%"},
    "speed": {"min": 950, "max": 1050, "unit": "RPM"},
    "torque": {"min": 45, "max": 75, "unit": "Nm"},
    "thickness": {"min": 9.8, "max": 10.2, "unit": "mm"},
    "surface_roughness": {"min": 0.0, "max": 1.6, "unit": "μm"},
}

# Defect Risk Levels
RISK_LEVELS = {
    "LOW": {"threshold": 0.2, "color": "green"},
    "MEDIUM": {"threshold": 0.5, "color": "orange"},
    "HIGH": {"threshold": 0.75, "color": "red"},
    "CRITICAL": {"threshold": 1.0, "color": "darkred"},
}

# Agent Configuration
AGENT_NAMES = {
    "monitor": "Process Monitoring Agent",
    "analysis": "Quality Analysis Agent",
    "prediction": "Defect Prediction Agent",
    "optimization": "Process Optimization Agent",
}

# Flask Server
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = False

# Data simulation interval (seconds)
MONITORING_INTERVAL = 5
DATA_HISTORY_SIZE = 100
