"""
IBM Granite AI Client — wraps ibm-watsonx-ai for text generation
"""

import json
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from config import IBM_API_KEY, IBM_PROJECT_ID, IBM_WATSONX_URL, GRANITE_MODEL_ID


class GraniteClient:
    """Singleton IBM Granite inference client."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        try:
            credentials = Credentials(
                url=IBM_WATSONX_URL,
                api_key=IBM_API_KEY,
            )
            self.model = ModelInference(
                model_id=GRANITE_MODEL_ID,
                credentials=credentials,
                project_id=IBM_PROJECT_ID,
                params={
                    GenParams.MAX_NEW_TOKENS: 1024,
                    GenParams.MIN_NEW_TOKENS: 50,
                    GenParams.TEMPERATURE: 0.3,
                    GenParams.TOP_P: 0.9,
                    GenParams.REPETITION_PENALTY: 1.1,
                },
            )
            self._initialized = True
            print("[GraniteClient] Connected to IBM Granite model successfully.")
        except Exception as exc:
            print(f"[GraniteClient] WARNING: Could not connect to IBM Granite: {exc}")
            self.model = None
            self._initialized = True

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate text from the Granite model."""
        if self.model is None:
            return self._fallback_response(prompt)
        try:
            response = self.model.generate_text(prompt=prompt)
            return response.strip() if isinstance(response, str) else str(response)
        except Exception as exc:
            print(f"[GraniteClient] Generation error: {exc}")
            return self._fallback_response(prompt)

    def _fallback_response(self, prompt: str) -> str:
        """Rule-based fallback when Granite is unavailable."""
        if "anomal" in prompt.lower() or "defect" in prompt.lower():
            return (
                "ANALYSIS: Parameter deviation detected. Recommend immediate inspection "
                "of temperature and pressure sensors. Adjust machine calibration and "
                "review material feed rates to restore quality within specification."
            )
        if "optim" in prompt.lower() or "recommend" in prompt.lower():
            return (
                "RECOMMENDATION: Reduce spindle speed by 5%, increase coolant flow by 10%, "
                "and recalibrate the pressure valve to maintain optimal operating range. "
                "Schedule preventive maintenance within 24 hours."
            )
        return (
            "INSIGHT: All monitored parameters are within acceptable ranges. "
            "Continue standard operating procedures and maintain current calibration schedule."
        )
