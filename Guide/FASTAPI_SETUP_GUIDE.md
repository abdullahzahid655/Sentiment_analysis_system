# 🚀 FastAPI + Joblib Setup Guide

## Overview

This guide shows how to deploy your sentiment analysis model as a **REST API** using FastAPI and Joblib.

---

## Step 1: Train & Save the Model

### In Your Jupyter Notebook:
```python
import joblib

# After training the production_pipeline:
joblib.dump(production_pipeline, 'sentiment_model.pkl')
print("✅ Model saved to sentiment_model.pkl")
```

**What gets saved?**
- TF-IDF Vectorizer (vocabulary + term weights)
- Logistic Regression model (word importance)

**File size**: Usually 5-50 MB depending on vocabulary size

---

## Step 2: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Or manually:
pip install fastapi uvicorn joblib scikit-learn pandas numpy nltk
```

---

## Step 3: Understand the API

### File: `sentiment_api.py`

**Key Components**:

1. **Request Models** (Data validation)
   ```python
   class SingleTextRequest(BaseModel):
       text: str
   ```
   → Ensures incoming data is valid JSON

2. **Response Models** (Output format)
   ```python
   class PredictionResult(BaseModel):
       sentiment: str
       confidence: float
       positive_prob: float
       negative_prob: float
   ```
   → Ensures consistent response format

3. **Load Model**
   ```python
   MODEL = joblib.load('sentiment_model.pkl')
   ```
   → Load pre-trained model from disk

4. **Endpoints** (API routes)
   ```python
   @app.post("/predict")
   async def predict(request: SingleTextRequest):
       # Make prediction
   ```

---

## Step 4: Run the API

### Option A: Direct Python
```bash
python sentiment_api.py
```

Output:
```
═══════════════════════════════════════════════════════════
🎬 Starting IMDB Sentiment Analyzer API
═══════════════════════════════════════════════════════════
📖 API Documentation (Interactive UI):
   http://localhost:8000/docs
═══════════════════════════════════════════════════════════
```

### Option B: Using Uvicorn
```bash
uvicorn sentiment_api:app --reload --port 8000
```

### Option C: Different Port
```bash
python sentiment_api.py  # Change port in the script
# or
uvicorn sentiment_api:app --port 9000
```

---

## Step 5: Test the API

### A. Interactive Documentation (Easiest!)

Open in your browser:
```
http://localhost:8000/docs
```

You'll see:
- All available endpoints
- Input/output specifications
- **"Try it out"** button to test directly

### B. Command Line (curl)

**Single Prediction**:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This movie is amazing!"}'
```

**Response**:
```json
{
  "text": "This movie is amazing!",
  "sentiment": "positive",
  "confidence": 0.9453,
  "positive_prob": 0.9453,
  "negative_prob": 0.0547,
  "timestamp": "2024-03-03T15:45:30.123456"
}
```

**Batch Prediction**:
```bash
curl -X POST "http://localhost:8000/batch_predict" \
     -H "Content-Type: application/json" \
     -d '{
       "texts": [
         "Great movie!",
         "Terrible waste of time",
         "Not bad"
       ]
     }'
```

**Health Check**:
```bash
curl http://localhost:8000/health
```

### C. Python Script

Create `test_api.py`:
```python
import requests

BASE_URL = "http://localhost:8000"

# Test single prediction
print("Testing single prediction...")
response = requests.post(
    f"{BASE_URL}/predict",
    json={"text": "I absolutely loved this movie!"}
)
print(response.json())
print()

# Test batch prediction
print("Testing batch prediction...")
response = requests.post(
    f"{BASE_URL}/batch_predict",
    json={
        "texts": [
            "Excellent film",
            "Terrible movie",
            "Pretty good"
        ]
    }
)
print(response.json())
print()

# Test health check
print("Testing health check...")
response = requests.get(f"{BASE_URL}/health")
print(response.json())
```

Run it:
```bash
python test_api.py
```

### D. JavaScript (From a Web Page)

```html
<!DOCTYPE html>
<html>
<head>
    <title>Sentiment API Test</title>
</head>
<body>
    <h1>Sentiment Analyzer</h1>
    <input type="text" id="reviewInput" placeholder="Enter a movie review...">
    <button onclick="predictSentiment()">Analyze</button>
    <pre id="result"></pre>

    <script>
        async function predictSentiment() {
            const text = document.getElementById('reviewInput').value;
            const response = await fetch('http://localhost:8000/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: text})
            });
            const data = await response.json();
            document.getElementById('result').textContent = JSON.stringify(data, null, 2);
        }
    </script>
</body>
</html>
```

---

## API Endpoints Reference

### 1. GET `/` (Welcome)
**Returns**: API information and available endpoints
```bash
curl http://localhost:8000/
```

### 2. GET `/health` (Health Check)
**Returns**: API status and model status
```bash
curl http://localhost:8000/health
```

### 3. POST `/predict` (Single Prediction)
**Input**: `{"text": "review text"}`
**Output**: Sentiment, confidence, probabilities
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Great movie!"}'
```

### 4. POST `/batch_predict` (Multiple Predictions)
**Input**: `{"texts": ["review1", "review2", ...]}`
**Output**: List of predictions with processing time
```bash
curl -X POST http://localhost:8000/batch_predict \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Good", "Bad"]}'
```

### 5. POST `/reload` (Reload Model)
**Use if**: You've retrained the model and saved a new version
```bash
curl -X POST http://localhost:8000/reload
```

---

## Understanding Responses

### Single Prediction Response
```json
{
  "text": "This movie is fantastic!",
  "sentiment": "positive",
  "confidence": 0.9523,
  "positive_prob": 0.9523,
  "negative_prob": 0.0477,
  "timestamp": "2024-03-03T15:45:30.123456"
}
```

**Fields**:
- `text`: Your input review
- `sentiment`: "positive" or "negative"
- `confidence`: How sure the model is (0-1 scale, 0.95 = 95%)
- `positive_prob`: Probability of positive class
- `negative_prob`: Probability of negative class (sums to 1.0)
- `timestamp`: ISO 8601 timestamp of when prediction was made

### Batch Prediction Response
```json
{
  "predictions": [
    {"text": "...", "sentiment": "positive", ...},
    {"text": "...", "sentiment": "negative", ...}
  ],
  "total": 2,
  "processing_time_ms": 45.3
}
```

---

## Error Handling

### Model Not Loaded
**Status**: 503 Service Unavailable
```json
{
  "detail": "Model not loaded. Please try again later."
}
```

**Solution**: Ensure `sentiment_model.pkl` exists in the working directory

### Invalid Input
**Status**: 422 Unprocessable Entity
```json
{
  "detail": [{
    "loc": ["body", "text"],
    "msg": "ensure this value has at least 1 characters",
    "type": "value_error.any.str.min_length"
  }]
}
```

**Solution**: Check your input format and values

---

## Production Deployment

### Option 1: Gunicorn (Recommended for Linux/Mac)

**Install**:
```bash
pip install gunicorn
```

**Run** (4 worker processes):
```bash
gunicorn -w 4 -b 0.0.0.0:8000 sentiment_api:app
```

### Option 2: Docker

**Create `Dockerfile`**:
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY sentiment_api.py .
COPY sentiment_model.pkl .

EXPOSE 8000

CMD ["python", "sentiment_api.py"]
```

**Build & Run**:
```bash
docker build -t sentiment-api .
docker run -p 8000:8000 sentiment-api
```

### Option 3: Heroku

**Create `Procfile`**:
```
web: gunicorn -b 0.0.0.0:$PORT sentiment_api:app
```

**Deploy**:
```bash
heroku create my-sentiment-api
git push heroku main
```

Access at: `https://my-sentiment-api.herokuapp.com/docs`

### Option 4: AWS Lambda + API Gateway

See AWS documentation on serverless FastAPI deployment.

### Option 5: Google Cloud Run

```bash
gcloud run deploy sentiment-api --source .
```

### Option 6: Microsoft Azure

```bash
az webapp up --name sentiment-api --resource-group myGroup
```

---

## Performance Optimization

### 1. Batch Processing
Instead of:
```python
# ❌ 100 individual requests (slow)
for review in reviews:
    requests.post("/predict", json={"text": review})
```

Do:
```python
# ✅ 1 batch request (faster)
requests.post("/batch_predict", json={"texts": reviews})
```

### 2. Caching
Add caching for identical predictions:
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def predict_cached(text):
    return predict_single(text)
```

### 3. Load Testing
Test performance with many concurrent requests:

**Using Apache Bench**:
```bash
ab -n 1000 -c 100 http://localhost:8000/health
```

**Using wrk**:
```bash
wrk -t12 -c400 -d30s http://localhost:8000/health
```

**Using Locust**:
```bash
pip install locust
locust -f locustfile.py --host=http://localhost:8000
```

### 4. Auto-scaling
In production, run multiple instances behind a load balancer:
```bash
# Terminal 1
python sentiment_api.py --port 8000

# Terminal 2
python sentiment_api.py --port 8001

# Terminal 3
python sentiment_api.py --port 8002

# Then use nginx/haproxy to load balance across ports
```

---

## Monitoring & Logging

### View Logs
The API logs all important events:
```
2024-03-03 15:45:30 INFO: ✅ Model loaded successfully
2024-03-03 15:45:31 INFO: POST /predict - sentiment: positive
2024-03-03 15:45:32 INFO: POST /batch_predict - 5 predictions
```

### Add Custom Logging
```python
import logging
logger = logging.getLogger(__name__)

logger.info(f"Prediction: {sentiment}")
logger.error(f"Model error: {e}")
```

### Monitor Response Times
```python
import time

start = time.time()
prediction = predict_single(text)
elapsed = time.time() - start
logger.info(f"Prediction took {elapsed:.3f}s")
```

---

## Troubleshooting

### Port Already in Use
```
OSError: [Errno 48] Address already in use
```

**Solution**:
```bash
# Kill process using port 8000
lsof -ti:8000 | xargs kill -9

# Or use a different port
uvicorn sentiment_api:app --port 9000
```

### Model Not Found
```
FileNotFoundError: [Errno 2] No such file or directory: 'sentiment_model.pkl'
```

**Solution**:
1. Train the model in the Jupyter notebook
2. Save it with: `joblib.dump(pipeline, 'sentiment_model.pkl')`
3. Ensure the file is in the same directory as `sentiment_api.py`

### CORS Issues (JavaScript Requests Fail)
```
Access to XMLHttpRequest blocked by CORS policy
```

**Solution**: The API already has CORS enabled in the code, but if you need to restrict it:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Restrict to your domain
)
```

### High Memory Usage
**Solution**: Implement request pooling and limit batch size
```python
MAX_BATCH_SIZE = 100
```

---

## Next Steps

1. ✅ Train the model (Jupyter notebook)
2. ✅ Save with joblib
3. ✅ Run FastAPI server
4. ✅ Test endpoints
5. ✅ Deploy to production
6. 📊 Monitor performance
7. 🔄 Retrain periodically with new data
8. 📈 Improve model accuracy

---

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Uvicorn Server](https://www.uvicorn.org/)
- [Joblib Documentation](https://joblib.readthedocs.io/)
- [REST API Best Practices](https://restfulapi.net/)
- [FastAPI Deployment Guide](https://fastapi.tiangolo.com/deployment/)

---

**Created**: March 2026  
**API Version**: 1.0.0  
**Status**: Production Ready
