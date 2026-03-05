# 🎨 Visual Guides & Diagrams

## 1. Complete Project Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    SENTIMENT ANALYSIS PIPELINE                   │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│   Raw Data       │
│   50k .txt files │
│   (aclImdb/)     │
└────────┬─────────┘
         │ STAGE 0: Load & Combine
         ▼
┌──────────────────┐
│   CSV File       │
│ (50k reviews)    │
│ (cleaned up)     │
└────────┬─────────┘
         │ STAGE 2: Explore
         ▼
┌──────────────────────┐
│  Data Insights       │
│  • Distribution      │
│  • Length stats      │
│  • Label balance     │
└────────┬─────────────┘
         │ STAGE 3: Preprocess
         ▼
┌─────────────────────────────────────────┐
│  Cleaned Text                           │
│  • Lowercase                            │
│  • Remove HTML & special chars          │
│  • Expand contractions                  │
│  • Tokenize & lemmatize                 │
│  • Remove stopwords (keep negations)    │
└────────┬────────────────────────────────┘
         │ STAGE 4: Feature Extraction
         ▼
┌──────────────────────────┐      ┌──────────────────────────┐
│   Bag of Words (BoW)     │      │    TF-IDF Vectors        │
│   Raw word counts        │      │    Weighted by rarity    │
└────────┬─────────────────┘      └────────┬─────────────────┘
         │                                 │
         └────────────┬────────────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │ Feature Matrix         │
         │ (40k x 10k sparse)     │
         │ Train: 80%, Test: 20%  │
         └────────────┬───────────┘
                      │ STAGE 5: Train Models
                      ▼
         ┌──────────────────────────────────┐
         │ 5 Model Configurations:          │
         │ 1. LogReg + TF-IDF    (92.1%) ⭐ │
         │ 2. LogReg + BoW       (91.5%)    │
         │ 3. CompNB + TF-IDF    (89.3%)    │
         │ 4. MNB + TF-IDF       (88.7%)    │
         │ 5. MNB + BoW          (87.9%)    │
         └────────────┬───────────────────┘
                      │ STAGE 6: Evaluate
                      ▼
         ┌──────────────────────────────────┐
         │ Evaluation Metrics:              │
         │ • Accuracy: 92.1%                │
         │ • Precision: 92%                 │
         │ • Recall: 91%                    │
         │ • F1: 91%                        │
         │ • ROC-AUC: 0.97                  │
         └────────────┬───────────────────┘
                      │ STAGE 7: Visualize
                      ▼
         ┌──────────────────────────────────┐
         │ 6 Charts:                        │
         │ • Model comparison               │
         │ • Confusion matrix               │
         │ • ROC curve                      │
         │ • Top words                      │
         │ • Word clouds                    │
         │ • Length distribution            │
         └────────────┬───────────────────┘
                      │ STAGE 8: Package
                      ▼
         ┌──────────────────────────────────┐
         │ Production Pipeline              │
         │ (TF-IDF + LogisticRegression)    │
         │ + Joblib (Model Persistence)     │
         └────────────┬───────────────────┘
                      │
                      ▼
         ┌──────────────────────────────────┐
         │ REST API (FastAPI)               │
         │ ├─ /predict (single)             │
         │ ├─ /batch_predict (multiple)     │
         │ ├─ /health (status)              │
         │ └─ /reload (refresh model)       │
         └────────────┬───────────────────┘
                      │
                      ▼
         ┌──────────────────────────────────┐
         │ Production Deployment            │
         │ ├─ Local                         │
         │ ├─ Docker                        │
         │ ├─ Cloud (AWS/Azure/GCP)         │
         │ └─ Web/Mobile Apps               │
         └──────────────────────────────────┘
```

---

## 2. Text Preprocessing Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                   TEXT PREPROCESSING STEPS                    │
└──────────────────────────────────────────────────────────────┘

STEP 1: LOWERCASE
━━━━━━━━━━━━━━━━━
Input:  "I REALLY Love This Movie!!!"
Output: "i really love this movie!!!"
Why:    Reduce vocabulary (Hello ≠ hello)

STEP 2: REMOVE HTML
━━━━━━━━━━━━━━━━━━━
Input:  "I really love <br /> this movie!!!"
Output: "I really love    this movie!!!"
Why:    IMDB data contains <br /> tags

STEP 3: EXPAND CONTRACTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:  "don't, I've, won't, isn't"
Output: "do not, I have, will not, is not"
Why:    Explicit forms are clearer

STEP 4: REMOVE SPECIAL CHARACTERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:  "I really love this movie!!! (10/10)"
Output: "I really love this movie      10 10"
Why:    Punctuation doesn't help sentiment

STEP 5: TOKENIZE (SPLIT INTO WORDS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:  "i really love this movie"
Output: ["i", "really", "love", "this", "movie"]
Why:    Process individual words

STEP 6: REMOVE STOPWORDS (BUT KEEP NEGATIONS!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Common words: "a", "the", "is", "and", "at"
Keep negations: "not", "no", "never", "neither"

Input:  ["i", "really", "love", "this", "movie"]
Output: ["really", "love", "movie"]
         (removed "i" and "this")

⚠️  CRITICAL EXAMPLE:
Input:  ["i", "do", "not", "like", "this", "movie"]
Output: ["not", "like", "movie"]  ← "not" is KEPT!
        (without "not": "like" = positive, but actually negative)

STEP 7: LEMMATIZE
━━━━━━━━━━━━━━━━
Input:  ["running", "runs", "really", "beautiful"]
Output: ["run", "run", "real", "beautiful"]
Why:    Same word in different forms = combine

FINAL OUTPUT:
═════════════════════════════════════════════
Original: "I really don't like this movie at all!!!"
Final:    ["really", "not", "like", "movie"]

Now we have:
✅ Lowercase
✅ No HTML
✅ No special chars
✅ Tokens
✅ Negations preserved
✅ Lemmatized
```

---

## 3. Feature Extraction Visualization

```
┌──────────────────────────────────────────────────────────────┐
│                   FEATURE EXTRACTION METHODS                 │
└──────────────────────────────────────────────────────────────┘

METHOD 1: BAG OF WORDS (BOW)
═════════════════════════════

Vocabulary: ["good", "bad", "movie", "like", "terrible"]

Review 1: "good movie good"
Count:    [2,      0,    1,      0,      0]
Vector:   [2, 0, 1, 0, 0]

Review 2: "bad movie terrible"
Count:    [0,   1,    0,      1,      1]
Vector:   [0, 1, 0, 1, 1]

Visualization:
            good bad movie like terrible
Review 1:    2   0   1     0    0      ← focuses on "good"
Review 2:    0   1   1     0    1      ← focuses on "bad"

Pros:  ✅ Simple, fast, interpretable
Cons:  ❌ All words treated equally
       ❌ Common words (movie) = rare words (mediocre)

─────────────────────────────────────────────────────────────

METHOD 2: TF-IDF (TERM FREQUENCY - INVERSE DOCUMENT FREQUENCY)
═══════════════════════════════════════════════════════════════

Formula: TF-IDF = (word frequency) × (rarity across corpus)

Example: In 1000 reviews,
- "movie" appears in 950 reviews → low weight (common)
- "masterpiece" appears in 20 reviews → high weight (rare)

Same reviews as above:

Review 1: "good movie good"
TF-IDF:   [0.85,  0.0,  0.15, 0.0, 0.0]
           (good is common, movie is very common)

Review 2: "bad movie terrible"
TF-IDF:   [0.0,  0.65, 0.15, 0.0, 0.75]
           (bad & terrible are important signals)

Visualization:
               good   bad   movie  like  terrible
Review 1:      0.85   0.0   0.15   0.0   0.0      ← high "good"
Review 2:      0.0    0.65  0.15   0.0   0.75     ← high "bad" & "terrible"

Pros:  ✅ Weights by importance
       ✅ Better model accuracy
       ✅ Ignores common filler words
Cons:  ❌ Slightly more complex

─────────────────────────────────────────────────────────────

SPARSE MATRIX (Memory Efficient)
════════════════════════════════

BoW typically creates SPARSE matrices (mostly zeros):

    ┌─────────────────────────────────────┐
    │ [2  0  1  0  0  0  0  0  0  0  ...] │
    │ [0  1  1  0  0  1  0  0  0  0  ...] │  95% zeros!
    │ [0  0  1  0  3  0  0  1  0  0  ...] │
    │ [1  0  0  2  0  0  1  0  0  0  ...] │
    └─────────────────────────────────────┘

Scikit-learn stores efficiently (only stores non-zero values)
So even with 10,000 vocabulary words, memory usage is manageable.
```

---

## 4. Model Training Process

```
┌──────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING VISUALIZED                 │
└──────────────────────────────────────────────────────────────┘

STEP 1: INITIALIZE
━━━━━━━━━━━━━━━━
Random weights:  w₁=0.02, w₂=-0.01, w₃=0.05, ...

STEP 2: MAKE PREDICTIONS
━━━━━━━━━━━━━━━━━━━━━━━
For "good movie":
  Score = w₁×2 + w₂×0 + w₃×1 + ...
        = 0.02×2 + (-0.01)×0 + 0.05×1
        = 0.04 + 0 + 0.05
        = 0.09

Convert to probability: sigmoid(0.09) ≈ 0.52 → POSITIVE (but barely!)
Actual label: POSITIVE (1)
Error: 0.52 - 1 = -0.48 (we were too confident it's negative)

STEP 3: UPDATE WEIGHTS (Use gradient descent)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Adjust weights to reduce error:
  w₁ = 0.02 + 0.01 = 0.03   (increase slightly)
  w₂ = -0.01 + 0.00 = -0.01 (no change, word isn't in review)
  w₃ = 0.05 + 0.01 = 0.06   (increase)

STEP 4: REPEAT
━━━━━━━━━━━━━━
After iteration 1: Error = 0.48
After iteration 2: Error = 0.45
After iteration 3: Error = 0.40
...
After iteration 100: Error = 0.01 ← Converged!

VISUALIZATION: Error Over Iterations
────────────────────────────────────

Error
 │     ╱╲
 │    ╱  ╲___
 │   ╱       ╲___
0.5│  ╱          ╲____
 │ ╱               ╲___
 │╱                    ╲__
0.0└──────────────────────────► Iterations
     0   25  50  75  100

Final weights learned by model:
═════════════════════════════════

Word       Weight   Interpretation
────────────────────────────────────
excellent   +0.85   Strongly positive
wonderful   +0.80   Strongly positive
good        +0.50   Positive
like        +0.30   Slightly positive
bad         -0.75   Strongly negative
terrible    -0.90   Very negative
hate        -0.85   Very negative
awful       -0.70   Negative
```

---

## 5. Model Evaluation Metrics

```
┌──────────────────────────────────────────────────────────────┐
│                    EVALUATION METRICS EXPLAINED               │
└──────────────────────────────────────────────────────────────┘

CONFUSION MATRIX
════════════════════════════════════════════════════════════════

                      Predicted
                    Positive  Negative
Actual  Positive   │ TP(850) │ FN(50)  │
        Negative   │ FP(70)  │ TN(930) │

TP = True Positive   (correctly predicted positive)  = 850
FP = False Positive  (wrongly predicted positive)     = 70
FN = False Negative  (wrongly predicted negative)     = 50
TN = True Negative   (correctly predicted negative)   = 930

TOTAL = 1,000 test samples

METRIC CALCULATIONS
═════════════════════════════════════════════════════════════════

1. ACCURACY = (TP + TN) / TOTAL
   = (850 + 930) / 1000
   = 1780 / 1000
   = 0.92 = 92%
   
   What it means: "Overall, how often is the model correct?"
   Use case: Balanced datasets (50/50 positive/negative)

─────────────────────────────────────────────────────────────

2. PRECISION = TP / (TP + FP)
   = 850 / (850 + 70)
   = 850 / 920
   = 0.92 = 92%
   
   What it means: "Of the positive predictions we made,
                   how many were actually positive?"
   Use case: False positives are expensive
            (e.g., flagging innocent reviews as spam)

─────────────────────────────────────────────────────────────

3. RECALL = TP / (TP + FN)
   = 850 / (850 + 50)
   = 850 / 900
   = 0.94 = 94%
   
   What it means: "Of all actual positive cases,
                   how many did we catch?"
   Use case: False negatives are expensive
            (e.g., missing actual positive reviews)

─────────────────────────────────────────────────────────────

4. F1 SCORE = 2 × (Precision × Recall) / (Precision + Recall)
   = 2 × (0.92 × 0.94) / (0.92 + 0.94)
   = 2 × 0.8648 / 1.86
   = 0.93 = 93%
   
   What it means: "Harmonic mean of precision & recall"
   Use case: When you care about BOTH false positives
            AND false negatives equally

─────────────────────────────────────────────────────────────

5. ROC-AUC SCORE
   
   ROC Curve: Plot of True Positive Rate vs False Positive Rate
   
   TPR (Sensitivity) = TP / (TP + FN) = Recall
   FPR (1 - Specificity) = FP / (FP + TN)
   
   ROC Curve Graph:
   ┌──────────────────────────┐
   │1.0┼─────────────────┐    │
   │   │                ╱     │ Random Classifier
   │0.8│              ╱       │ (diagonal)
   │   │            ╱         │
   │TPR│          ╱           │
   │0.6│        ╱             │ Your Model
   │   │      ╱╱              │ (curved)
   │0.4│    ╱╱                │
   │   │  ╱╱                  │
   │0.2│╱╱                    │
   │   │╱                     │
   │0.0└──────────────────────┤
   │   0.0  0.2  0.4  0.6  0.8  1.0
   │                FPR
   └──────────────────────────┘
   
   AUC = Area Under the Curve
   
   0.5 = random guessing (diagonal line)
   1.0 = perfect classifier
   0.97 = excellent (our model) ⭐
   
   What it means: "Overall ability to distinguish between classes"
   Use case: Works for imbalanced data, threshold-independent

INTERPRETATION GUIDE
═════════════════════════════════════════════════════════════════

┌─────────────────┬─────────────────────────────────────────────┐
│ Metric          │ Interpretation                              │
├─────────────────┼─────────────────────────────────────────────┤
│ Accuracy: 92%   │ Model is correct 92% of the time ✅          │
│ Precision: 92%  │ When we say positive, 92% are actually ✅    │
│ Recall: 94%     │ We catch 94% of actual positives ✅          │
│ F1: 93%         │ Good balance of precision & recall ✅        │
│ ROC-AUC: 0.97   │ Excellent discrimination ability ✅          │
└─────────────────┴─────────────────────────────────────────────┘
```

---

## 6. API Request-Response Flow

```
┌──────────────────────────────────────────────────────────────┐
│                    API REQUEST-RESPONSE FLOW                 │
└──────────────────────────────────────────────────────────────┘

CLIENT SIDE (Your Code)
═══════════════════════════════════════════════════════════════

┌────────────────────────────────────────┐
│  import requests                       │
│                                        │
│  review = "This movie is amazing!"   │
│                                        │
│  response = requests.post(             │
│    "http://localhost:8000/predict",   │
│    json={"text": review}              │
│  )                                     │
│                                        │
│  data = response.json()                │
│  print(data['sentiment'])              │
└─────────────┬──────────────────────────┘
              │
              │ HTTP POST Request
              │ ┌──────────────────────────────┐
              │ │ POST /predict HTTP/1.1       │
              │ │ Host: localhost:8000         │
              │ │ Content-Type: application/   │
              │ │   json                       │
              │ │                              │
              │ │ {"text": "This movie is      │
              │ │   amazing!"}                 │
              │ └──────────────────────────────┘
              │
              ▼

SERVER SIDE (FastAPI)
═══════════════════════════════════════════════════════════════

              ┌──────────────────────────────┐
              │ FastAPI Receives Request     │
              │ • Parses JSON                │
              │ • Validates input            │
              │ • Routes to endpoint         │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │ @app.post("/predict")        │
              │ async def predict(request)   │
              │ {                            │
              │   text = request.text        │
              │   ...                        │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │ Load Pre-trained Model       │
              │ • TF-IDF Vectorizer          │
              │ • Logistic Regression        │
              │                              │
              │ model.predict([text])        │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │ Generate Response            │
              │ {                            │
              │   "sentiment": "positive",   │
              │   "confidence": 0.94,        │
              │   ...                        │
              │ }                            │
              └──────────────┬───────────────┘
                             │
                             │ HTTP Response
                             │ ┌──────────────────────────────┐
                             │ │ HTTP/1.1 200 OK              │
                             │ │ Content-Type: application/   │
                             │ │   json                       │
                             │ │ Content-Length: 120          │
                             │ │                              │
                             │ │ {                            │
                             │ │   "text": "This movie is     │
                             │ │     amazing!",               │
                             │ │   "sentiment": "positive",   │
                             │ │   "confidence": 0.9453,      │
                             │ │   "positive_prob": 0.9453,   │
                             │ │   "negative_prob": 0.0547,   │
                             │ │   "timestamp": "2024-03..."  │
                             │ │ }                            │
                             │ └──────────────────────────────┘
                             │
                             ▼

CLIENT SIDE (continued)
═══════════════════════════════════════════════════════════════

              ┌──────────────────────────────┐
              │ Parse Response               │
              │ {                            │
              │   "sentiment": "positive"    │
              │   "confidence": 0.9453       │
              │ }                            │
              │                              │
              │ Output: POSITIVE (94.53%)    │
              └──────────────────────────────┘

BATCH REQUEST FLOW
═══════════════════════════════════════════════════════════════

Input:  3 reviews
        ├─ "Great movie!"
        ├─ "Terrible waste of time"
        └─ "Not bad"

        │
        ▼

Request:
        POST /batch_predict
        {
          "texts": [
            "Great movie!",
            "Terrible waste of time",
            "Not bad"
          ]
        }

        │ Processing
        ▼

Server:
        For each review:
        1. Vectorize text (TF-IDF)
        2. Make prediction
        3. Get confidence
        4. Add to results

        │
        ▼

Response:
        {
          "predictions": [
            {"text": "Great movie!", "sentiment": "positive", ...},
            {"text": "Terrible...", "sentiment": "negative", ...},
            {"text": "Not bad", "sentiment": "positive", ...}
          ],
          "total": 3,
          "processing_time_ms": 45.3
        }

        │
        ▼

Output:
        ✅ Great movie! → POSITIVE
        ❌ Terrible waste of time → NEGATIVE
        ✅ Not bad → POSITIVE
```

---

## 7. Deployment Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                  DEPLOYMENT OPTIONS VISUALIZATION            │
└──────────────────────────────────────────────────────────────┘

OPTION 1: LOCAL DEVELOPMENT
════════════════════════════════════════════════════════════════

Your Computer
    │
    ├─ sentiment_api.py (FastAPI server)
    ├─ sentiment_model.pkl (ML model)
    └─ requirements.txt (dependencies)
    
Access: http://localhost:8000

Pros:  ✅ Easy testing & development
Cons:  ❌ Only you can access
       ❌ Down when you close it

─────────────────────────────────────────────────────────────

OPTION 2: DOCKER CONTAINERIZATION
════════════════════════════════════════════════════════════════

Docker Container
┌────────────────────────────┐
│ sentiment_api.py           │
│ sentiment_model.pkl        │
│ requirements.txt           │
│ Python + Dependencies      │
│ Uvicorn Server             │
│ Port 8000                  │
└────────────────────────────┘
     │
     ▼
Access from any machine: http://ip:8000

Pros:  ✅ Reproducible across machines
       ✅ Easy to scale
       ✅ Production-ready
Cons:  ❌ Requires Docker installation

─────────────────────────────────────────────────────────────

OPTION 3: CLOUD DEPLOYMENT (AWS, Azure, Google Cloud)
════════════════════════════════════════════════════════════════

Your Local Code
    │
    ├─ Push to GitHub
    │
    ▼
    │
    ├─ GitHub → Cloud Provider
    │          (automatic)
    │
    ▼
    
Cloud Infrastructure:

Load Balancer
    │
    ├─ API Instance 1 (Port 8000)
    ├─ API Instance 2 (Port 8000)
    └─ API Instance 3 (Port 8000)
    
Each Instance:
    ├─ sentiment_api.py
    ├─ sentiment_model.pkl
    └─ Auto-scale based on demand

Access: https://api.yourdomain.com/predict

Benefits:
✅ Highly available (multiple instances)
✅ Auto-scaling (handles traffic spikes)
✅ HTTPS (secure)
✅ Global CDN (fast worldwide)
✅ Monitoring & logging
✅ Automatic backups

─────────────────────────────────────────────────────────────

OPTION 4: SERVERLESS (AWS Lambda, Google Cloud Functions)
════════════════════════════════════════════════════════════════

Your Code
    │
    ▼
Upload to Cloud Provider
    │
    ▼
Function runs on-demand:

    Request 1 ──┐
    Request 2 ──┼──→ [Spin up container]
    Request 3 ──┤    [Execute function]
    Request 4 ──┤    [Return response]
    Request 5 ──┘    [Shut down if idle]
    
Pay only for what you use!

Benefits:
✅ No servers to manage
✅ Auto-scaling by default
✅ Cost-effective
✅ Minimal configuration

Cons:
❌ Cold start latency (first request slow)
❌ Limited execution time
```

---

## 8. Model Decision Tree

```
WHICH MODEL SHOULD I CHOOSE?
════════════════════════════════════════════════════════════════

                          START
                            │
                            ▼
                     Need interpretability?
                      /             \
                    YES              NO
                    │                 │
                    ▼                 ▼
          Logistic Regression    LSTM / BERT?
          + TF-IDF               (Deep Learning)
          ✓ Fast                 
          ✓ See which words       ✓ Better accuracy
            matter                ✓ Handles context
          ✓ 92% accuracy          ✗ Black box
                                  ✗ Slow/expensive
                    │                 │
                    └─────────┬────────┘
                              │
                              ▼
                         Dataset size?
                        /            \
                    Small            Large
                  (<10k)            (>100k)
                    │                 │
                    ▼                 ▼
         Logistic Regression    LSTM / BERT
         Naive Bayes            Deep Learning
         (enough data)
                    │                 │
                    └─────────┬────────┘
                              │
                              ▼
                      Performance OK?
                        /          \
                      YES          NO
                       │            │
                       ▼            ▼
                    Deploy    Improve Model:
                    ✅        1. More data
                              2. Better preprocessing
                              3. Hyperparameter tune
                              4. Ensemble methods
                              5. Deep learning
```

---

**These diagrams help visualize:**
- The complete pipeline flow
- How preprocessing transforms data
- Feature extraction methods
- Model training process
- Evaluation metrics
- API communication
- Deployment options
