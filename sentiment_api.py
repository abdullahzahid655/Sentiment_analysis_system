"""
🚀 FastAPI Sentiment Analysis REST API
=====================================

A production-ready REST API for IMDB sentiment analysis using:
- FastAPI: Modern, fast web framework
- Joblib: Load/save ML models
- Scikit-learn: Pre-trained sentiment classifier

Installation:
    pip install fastapi uvicorn joblib scikit-learn pandas numpy

Running:
    python sentiment_api.py
    OR
    uvicorn sentiment_api:app --reload --port 8000

Testing:
    Open: http://localhost:8000/docs (Interactive Swagger UI)
    Or:   http://localhost:8000/redoc (Alternative UI)

Example Requests:
    # Single prediction
    curl -X POST "http://localhost:8000/predict" \
         -H "Content-Type: application/json" \
         -d '{"text": "This movie is absolutely fantastic!"}'

    # Batch prediction
    curl -X POST "http://localhost:8000/batch_predict" \
         -H "Content-Type: application/json" \
         -d '{"texts": ["Great movie", "Terrible film"]}'

    # Health check
    curl http://localhost:8000/health
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field, ConfigDict
import joblib
import numpy as np
import os
from typing import List, Optional
import logging
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
# ⚙️ LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# 📱 FASTAPI APP INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="🎬 IMDB Sentiment Analyzer API",
    description="Production-grade API for classifying movie review sentiment (positive/negative)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# ── CORS Configuration ────────────────────────────────────────────────────────
# Allows requests from web browsers/other domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════════════
# 📦 REQUEST/RESPONSE MODELS (Data Validation)
# ═══════════════════════════════════════════════════════════════════════════════

class SingleTextRequest(BaseModel):
    """Schema for single text prediction request."""
    text: str = Field(..., min_length=1, max_length=5000, description="Movie review text to analyze")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "This movie is absolutely fantastic and I loved every minute!"
            }
        }
    )


class BatchTextRequest(BaseModel):
    """Schema for batch prediction requests."""
    texts: List[str] = Field(..., min_length=1, max_length=100, description="List of reviews")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "texts": [
                    "Great movie!",
                    "Absolutely terrible",
                    "Not bad, quite enjoyable"
                ]
            }
        }
    )


class PredictionResult(BaseModel):
    """Schema for single prediction result."""
    text: str = Field(description="The input text")
    sentiment: str = Field(description="Predicted sentiment: 'positive' or 'negative'")
    confidence: float = Field(description="Confidence score (0-1)")
    positive_prob: float = Field(description="Probability of positive class (0-1)")
    negative_prob: float = Field(description="Probability of negative class (0-1)")
    timestamp: str = Field(description="When prediction was made (ISO 8601 format)")


class BatchPredictionResult(BaseModel):
    """Schema for batch predictions."""
    predictions: List[PredictionResult]
    total: int = Field(description="Total number of predictions")
    processing_time_ms: float = Field(description="Time to process all requests (milliseconds)")


class HealthCheckResponse(BaseModel):
    """Schema for health check response."""
    status: str
    message: str
    model_loaded: bool
    timestamp: str


# ═══════════════════════════════════════════════════════════════════════════════
# 🤖 MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_PATH = "sentiment_model.pkl"
MODEL = None
MODEL_LOADED = False

def load_model():
    """Load the pre-trained sentiment model from disk."""
    global MODEL, MODEL_LOADED
    
    try:
        if os.path.exists(MODEL_PATH):
            MODEL = joblib.load(MODEL_PATH)
            MODEL_LOADED = True
            logger.info(f"✅ Model loaded successfully from {MODEL_PATH}")
        else:
            logger.warning(f"⚠️  Model file not found at {MODEL_PATH}")
            logger.info("   To use the API, please:")
            logger.info("   1. Run the Jupyter notebook to train the model")
            logger.info("   2. Save it with: joblib.dump(production_pipeline, 'sentiment_model.pkl')")
            MODEL_LOADED = False
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        MODEL_LOADED = False

# Load model on startup
load_model()

# ═══════════════════════════════════════════════════════════════════════════════
# 🔍 PREDICTION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def predict_single(text: str) -> dict:
    """
    Make a prediction for a single text.
    
    Args:
        text: Review text
        
    Returns:
        Dictionary with sentiment prediction and probabilities
        
    Raises:
        ValueError: If model is not loaded
    """
    if not MODEL_LOADED or MODEL is None:
        raise ValueError("Model not loaded. Please ensure sentiment_model.pkl exists.")
    
    # Get prediction and probabilities
    prediction = MODEL.predict([text])[0]
    probabilities = MODEL.predict_proba([text])[0]
    
    # Map to sentiment labels
    sentiment = "positive" if prediction == 1 else "negative"
    confidence = float(probabilities.max())
    positive_prob = float(probabilities[1])  # Probability of positive class
    negative_prob = float(probabilities[0])  # Probability of negative class
    
    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 4),
        "positive_prob": round(positive_prob, 4),
        "negative_prob": round(negative_prob, 4),
    }

# ═══════════════════════════════════════════════════════════════════════════════
# 🌐 API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Info"])
async def root():
    """
    Redirect to the web interface.
    """
    return FileResponse("index.html")


@app.get("/ui", response_class=HTMLResponse, tags=["Info"])
async def web_ui():
    """
    Serve the professional web UI.
    """
    try:
        with open("index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(
            """
            <html>
            <body style="font-family: Arial; text-align: center; padding: 50px;">
            <h1>⚠️ Web UI Not Found</h1>
            <p>The index.html file is missing. Please create it.</p>
            </body>
            </html>
            """,
            status_code=404
        )


@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Returns API and model status.
    
    Use this endpoint to verify the API is running and the model is loaded.
    """
    return HealthCheckResponse(
        status="healthy" if MODEL_LOADED else "degraded",
        message="Model is ready for predictions" if MODEL_LOADED else "Model not loaded",
        model_loaded=MODEL_LOADED,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict", response_model=PredictionResult, tags=["Predictions"])
async def predict(request: SingleTextRequest):
    """
    Predict sentiment for a single movie review.
    
    **Input**: Movie review text (up to 5000 characters)
    
    **Output**: 
    - sentiment: "positive" or "negative"
    - confidence: How sure the model is (0.0-1.0)
    - positive_prob: Probability of positive class
    - negative_prob: Probability of negative class
    - timestamp: When the prediction was made
    
    **Example**:
    ```
    {
        "text": "This movie is absolutely fantastic!"
    }
    ```
    
    **Response**:
    ```
    {
        "text": "This movie is absolutely fantastic!",
        "sentiment": "positive",
        "confidence": 0.9523,
        "positive_prob": 0.9523,
        "negative_prob": 0.0477,
        "timestamp": "2024-03-03T15:45:30.123456"
    }
    ```
    """
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )
    
    try:
        prediction = predict_single(request.text)
        
        return PredictionResult(
            text=request.text,
            sentiment=prediction["sentiment"],
            confidence=prediction["confidence"],
            positive_prob=prediction["positive_prob"],
            negative_prob=prediction["negative_prob"],
            timestamp=datetime.now().isoformat(),
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict", response_model=BatchPredictionResult, tags=["Predictions"])
async def batch_predict(request: BatchTextRequest):
    """
    Predict sentiment for multiple reviews at once.
    
    **Useful for**:
    - Analyzing multiple reviews in one request
    - Monitoring feedback/comments
    - Batch processing existing reviews
    
    **Input**: List of review texts (1-100 texts)
    
    **Output**: List of predictions with processing time
    
    **Example**:
    ```
    {
        "texts": [
            "Great movie, loved it!",
            "Absolutely terrible, waste of time",
            "Pretty good, but had some issues"
        ]
    }
    ```
    
    **Response**:
    ```
    {
        "predictions": [
            {"text": "Great movie...", "sentiment": "positive", ...},
            {"text": "Absolutely terrible...", "sentiment": "negative", ...},
            {"text": "Pretty good...", "sentiment": "positive", ...}
        ],
        "total": 3,
        "processing_time_ms": 45.3
    }
    ```
    """
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )
    
    import time
    start_time = time.time()
    
    try:
        predictions = []
        
        for text in request.texts:
            prediction = predict_single(text)
            predictions.append(
                PredictionResult(
                    text=text,
                    sentiment=prediction["sentiment"],
                    confidence=prediction["confidence"],
                    positive_prob=prediction["positive_prob"],
                    negative_prob=prediction["negative_prob"],
                    timestamp=datetime.now().isoformat(),
                )
            )
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return BatchPredictionResult(
            predictions=predictions,
            total=len(predictions),
            processing_time_ms=round(processing_time_ms, 2),
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/reload", tags=["Admin"])
async def reload_model():
    """
    Reload the model from disk.
    
    Use this if you've retrained the model and saved a new version.
    """
    load_model()
    
    if MODEL_LOADED:
        return {
            "status": "success",
            "message": "Model reloaded successfully",
            "model_loaded": True,
            "timestamp": datetime.now().isoformat(),
        }
    else:
        raise HTTPException(
            status_code=500,
            detail="Failed to reload model"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 RUNNING THE SERVER
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    
    logger.info("═" * 70)
    logger.info("🎬 Starting IMDB Sentiment Analyzer API")
    logger.info("═" * 70)
    
    if not MODEL_LOADED:
        logger.warning("⚠️  WARNING: Model not loaded!")
        logger.warning("   The API will return 503 (Service Unavailable) until you train and save the model.")
        logger.warning("")
        logger.warning("   To fix this:")
        logger.warning("   1. Run the Jupyter notebook: sentiment_analysis_nlp.ipynb")
        logger.warning("   2. Train the model and save it with:")
        logger.warning("      joblib.dump(production_pipeline, 'sentiment_model.pkl')")
    
    logger.info("")
    logger.info("📖 API Documentation (Interactive UI):")
    logger.info("   http://localhost:8000/docs")
    logger.info("")
    logger.info("📋 Alternative Documentation:")
    logger.info("   http://localhost:8000/redoc")
    logger.info("")
    logger.info("💡 Testing the API:")
    logger.info("   http://localhost:8000/health  (Check status)")
    logger.info("═" * 70)
    
    uvicorn.run(
        app,
        host="0.0.0.0",  # Listen on all interfaces
        port=8000,       # Port 8000
        log_level="info",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 📚 USAGE EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════════
"""

🔹 CURL EXAMPLES (Command Line)
═════════════════════════════════

1. Single Prediction:
   ─────────────────
   curl -X POST "http://localhost:8000/predict" \
        -H "Content-Type: application/json" \
        -d '{"text": "This movie is absolutely fantastic!"}'

2. Batch Prediction:
   ─────────────────
   curl -X POST "http://localhost:8000/batch_predict" \
        -H "Content-Type: application/json" \
        -d '{
          "texts": [
            "Great movie!",
            "Terrible waste of time",
            "Not bad"
          ]
        }'

3. Health Check:
   ──────────────
   curl http://localhost:8000/health

4. Reload Model:
   ──────────────
   curl -X POST http://localhost:8000/reload


🔹 PYTHON EXAMPLES
════════════════

import requests

BASE_URL = "http://localhost:8000"

# Single Prediction
response = requests.post(
    f"{BASE_URL}/predict",
    json={"text": "I loved this movie!"}
)
print(response.json())

# Batch Prediction
response = requests.post(
    f"{BASE_URL}/batch_predict",
    json={
        "texts": [
            "Great film",
            "Terrible movie",
            "Average at best"
        ]
    }
)
print(response.json())

# Health Check
response = requests.get(f"{BASE_URL}/health")
print(response.json())


🔹 JAVASCRIPT EXAMPLES
═══════════════════

// Single Prediction
fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text: 'Excellent movie!'})
})
.then(r => r.json())
.then(data => console.log(data))

// Batch Prediction
fetch('http://localhost:8000/batch_predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        texts: ['Great!', 'Terrible!', 'OK...']
    })
})
.then(r => r.json())
.then(data => console.log(data))


🔹 DEPLOYMENT OPTIONS
═════════════════════

Local Development:
  python sentiment_api.py

Docker:
  docker build -t sentiment-api .
  docker run -p 8000:8000 sentiment-api

Heroku:
  heroku create my-sentiment-api
  git push heroku main

AWS Lambda (Serverless):
  pip install zappa
  zappa init
  zappa deploy dev

Google Cloud Run:
  gcloud run deploy sentiment-api --source .

AWS EC2:
  scp sentiment_api.py ubuntu@ec2-instance:~
  ssh ubuntu@ec2-instance
  python sentiment_api.py

Azure:
  az webapp up --name sentiment-api --resource-group myGroup


🔹 PERFORMANCE TIPS
═══════════════════

1. **Batch Processing**: Use /batch_predict for multiple texts
   - Much faster than making multiple single requests
   - Example: 100 reviews in 1 request vs 100 requests

2. **Caching**: Store predictions for identical reviews
   - Reduces unnecessary computation
   - Improves response time

3. **Load Testing**: Test with many concurrent requests
   - Use tools like: Apache Bench, wrk, Locust
   - Example: ab -n 1000 -c 100 http://localhost:8000/health

4. **Auto-scaling**: In production, use load balancers
   - Multiple API instances
   - Distribute requests evenly

"""
