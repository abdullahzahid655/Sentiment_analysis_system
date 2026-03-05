# 🎬 Deep Explanation: IMDB Sentiment Analysis NLP Project

---

## 📑 Table of Contents
1. [Project Overview](#project-overview)
2. [Stage-by-Stage Breakdown](#stage-by-stage-breakdown)
3. [Key Concepts Explained](#key-concepts-explained)
4. [How Everything Works Together](#how-everything-works-together)
5. [FastAPI + Joblib: Production Deployment](#fastapi--joblib-production-deployment)

---

## Project Overview

### What is This Project?
This is a **Sentiment Analysis** system that reads movie reviews from IMDB and classifies them as either **Positive** or **Negative**.

### Key Numbers
- **Dataset**: 50,000 reviews (25,000 positive, 25,000 negative)
- **Source**: IMDB aclImdb folder (raw `.txt` files)
- **Goal**: Build a machine learning model that predicts if a review is positive or negative

### Real-World Application
Imagine you have a website with thousands of customer reviews. You want to automatically:
- Flag negative reviews for customer support
- Identify positive testimonials for marketing
- Monitor customer sentiment trends

This project shows exactly how to do that!

---

## Stage-by-Stage Breakdown

### **Stage 0: Load Raw Dataset from Folder Structure**

#### What's Happening?
The IMDB dataset comes as **50,000 individual `.txt` files** organized in folders:
```
aclImdb/
├── train/
│   ├── pos/ (12,500 positive reviews as .txt files)
│   ├── neg/ (12,500 negative reviews as .txt files)
│   └── unsup/ (50,000 unlabeled reviews — we skip these)
└── test/
    ├── pos/ (12,500 positive reviews)
    └── neg/ (12,500 negative reviews)
```

#### Key Concept: Why Do This?
- **Raw data** is messy and scattered across many files
- **Goal**: Combine everything into one clean CSV file so we can work with it easily in pandas

#### The Process
```python
1. Loop through train/pos, train/neg, test/pos, test/neg folders
2. Read each .txt file as a review
3. Extract the rating from the filename (e.g., "1234_8.txt" → rating 8)
4. Create a DataFrame with columns: [text, sentiment, split, rating, filename]
5. Save to CSV for future use (so you don't have to do this again!)
```

#### Why Separate Train/Test Folders?
- IMDB originally split their data 50/50 to help researchers
- We **combine them together** and create our own 80/20 split
- This gives our models **more training data** and ensures consistency

---

### **Stage 1: Imports & Configuration**

#### What's Happening?
Loading all the tools we need.

#### Key Libraries Explained

| Library | Purpose | Used For |
|---------|---------|----------|
| **pandas** | Data manipulation | Loading/storing CSV, filtering data |
| **numpy** | Numerical operations | Math operations on arrays |
| **nltk** | Natural Language Toolkit | Tokenization, stopwords, lemmatization |
| **scikit-learn** | Machine Learning | Vectorizers, classifiers, metrics |
| **matplotlib/seaborn** | Visualization | Creating charts and graphs |
| **wordcloud** | Visual word analysis | Creating word clouds |

#### Configuration Variables
```python
MAX_FEATURES = 10_000      # Keep only top 10k most important words
NGRAM_RANGE = (1, 2)       # Capture single words + bigrams ("not good")
TEST_SIZE = 0.2            # Use 20% of data for testing
RANDOM_STATE = 42          # Ensures reproducibility (same random numbers every run)
```

**Key Insight**: `RANDOM_STATE = 42` is a joke in the ML community (reference to Hitchhiker's Guide), but it's crucial because it makes experiments **reproducible** — everyone gets the same results.

---

### **Stage 2: Dataset Exploration (EDA)**

#### What's Happening?
Looking at the data to understand it before modeling.

#### Key Statistics
```
Total reviews: 50,000
Positive: 25,000 (50%)
Negative: 25,000 (50%)
Average review length: ~200 words
```

#### Important Observation: Balanced Dataset
- Each class (positive/negative) has **exactly 50%** of the data
- This is **excellent for training** — no class imbalance problems
- Real-world data is often imbalanced (90% negative, 10% positive), which makes training harder

#### What We Learn
- Review lengths vary widely (50–800 words)
- Some reviews are very short, others are essays
- This variation affects model performance (longer reviews = more signal)

---

### **Stage 3: Text Preprocessing Pipeline**

#### ⚠️ **Critical Stage** — This is where the magic happens!

Raw IMDB reviews contain **tons of noise**:
```
Original text:
"I really don't like it. It's just <br /> terrible and badly done!!!"

Processed text:
"really not like terribly badly do"
```

#### The 7-Step Cleaning Pipeline

**Step 1: Lowercase**
```python
"HeLLo" → "hello"
Reason: "Hello" and "hello" are the same word but appear different to computers
```

**Step 2: Remove HTML Tags**
```python
"I love <br /> movies" → "I love movies"
Reason: IMDB reviews often have <br /> tags (line breaks) that add no meaning
```

**Step 3: Expand Contractions**
```python
"don't" → "do not"
"I've" → "I have"
"won't" → "will not"
Reason: Contractions are shorthand. Expansion gives the model more explicit signals
```

**Step 4: Remove Special Characters**
```python
"Hello!!! Why??? It's 10/10!!!" → "Hello Why Its 10 10"
Reason: Punctuation and numbers don't usually help with sentiment
```

**Step 5: Tokenization (Split into Words)**
```python
"I love this movie" → ["I", "love", "this", "movie"]
Reason: Process individual words instead of sentences
```

**Step 6: Remove Stopwords (BUT Keep Negations!)**
```python
Remove: "the", "is", "at", "and", "a"
KEEP: "not", "no", "never", "neither"

Original: "This is not good"
After:    "not good"  ← "not" is crucial! Without it, we lose the negative meaning
```

**Stopwords** are common filler words that don't carry sentiment. **But we must keep negations** because:
- "not good" ≠ "good"
- "never bad" is POSITIVE (double negative)
- Without negations, we destroy meaning!

**Step 7: Lemmatization**
```python
"running", "runs", "ran" → "run"
"beautiful", "beautifully", "beautify" → "beautiful"
Reason: Same word in different forms = different features. Combine them into the base form.
```

#### Example Transformation
```
Input:  "I really don't think this movie is good, but I liked the acting!!!"
Output: "really not think movie good like act"
```

#### Why This Matters
Raw text is **too noisy** for machine learning. Preprocessing reduces noise and focuses on meaningful signals.

---

### **Stage 4: Feature Extraction**

#### The Core Problem
**ML models work with numbers, not words.** We need to convert text to vectors.

### Approach 1: Bag of Words (BoW)

**Concept**: Count how many times each word appears in each review.

**Example**:
```
Vocabulary: ["good", "bad", "movie", "like"]

Review 1: "good movie good"
Vector:   [2, 0, 1, 0]  ← "good" appears 2x, "bad" 0x, "movie" 1x, "like" 0x

Review 2: "bad movie bad like"
Vector:   [0, 2, 1, 1]
```

**Pros**:
- Simple and fast
- Interpretable
- Good baseline

**Cons**:
- Treats all words equally (common words like "movie" same as rare words like "mediocre")
- Ignores word importance

### Approach 2: TF-IDF (Term Frequency-Inverse Document Frequency)

**Better idea**: Give MORE weight to rare, informative words.

**Formula** (simplified):
```
TF-IDF = (frequency of word in document) × (rarity of word in corpus)

Example:
- "movie" appears in 80% of reviews → low weight
- "mediocre" appears in 2% of reviews → high weight
```

**Why This Matters**:
```
BoW vectors:      [0, 3, 5, 1, 0, 0, ...]  (all raw counts)
TF-IDF vectors:   [0, 0.2, 0.8, 0.5, 0, 0, ...]  (weighted by importance)
```

The second vector emphasizes INFORMATIVE words.

#### Configuration
```python
max_features=10_000      # Keep only top 10,000 words
ngram_range=(1,2)        # (1,2) = single words + 2-word phrases
min_df=2                 # Word must appear in at least 2 documents
max_df=0.95              # Word can appear in at most 95% of documents
sublinear_tf=True        # Use log(1+tf) instead of raw counts
```

**Why ngram_range=(1,2)?**
```
Single words: "bad", "movie"
Bigrams:      "not good", "very bad", "must see"

Bigrams capture sentiment better!
"not" alone is neutral
"not good" is negative
"not bad" is positive
```

---

### **Stage 5: Model Training**

#### What Are We Building?
5 different machine learning models to find the best one:

| Model | How It Works |
|-------|-------------|
| **Logistic Regression + TF-IDF** | Linear classifier: assigns weights to words. Most interpretable. |
| **Logistic Regression + BoW** | Same as above but with raw word counts instead. |
| **Multinomial Naive Bayes + TF-IDF** | Probabilistic classifier: assumes word independence. Fast. |
| **Multinomial Naive Bayes + BoW** | Probability-based with word counts. |
| **Complement Naive Bayes + TF-IDF** | Improved Naive Bayes variant. Better for balanced datasets. |

#### Understanding Logistic Regression

**Concept**: Assign a weight to each word.

```
Positive words get high weights:
- "masterpiece" → +0.85
- "excellent" → +0.78
- "love" → +0.62

Negative words get low weights:
- "terrible" → -0.92
- "awful" → -0.85
- "hate" → -0.70

Prediction for "This movie is excellent but terrible":
= 1×(0.78) + 1×(-0.92)
= -0.14 → NEGATIVE
```

Why this model?
- **Interpretable**: We can see which words matter
- **Fast**: Linear computation
- **Effective**: Great performance on this task

#### Understanding Naive Bayes

**Concept**: Calculate probability using word frequencies.

```
P(Positive | words) = P(words | Positive) × P(Positive) / P(words)

If a word appears frequently in positive reviews → higher probability
If a word appears frequently in negative reviews → lower probability
```

Why this model?
- **Probabilistic**: Gives confidence scores
- **Fast**: No gradient descent needed
- **Good baseline**: Works well for text classification

#### Training Process
```python
1. model = LogisticRegression()
2. model.fit(X_train, y_train)
   → Algorithm learns which words predict positive/negative
3. y_pred = model.predict(X_test)
   → Make predictions on unseen data
4. Evaluate accuracy, precision, recall, ROC-AUC
```

#### How Does the Model Learn?
**Logistic Regression uses calculus** to find optimal word weights:
1. Start with random weights
2. Make predictions on training data
3. Calculate error (how wrong are we?)
4. Adjust weights to reduce error
5. Repeat until weights stabilize

---

### **Stage 6: Evaluation & Metrics**

#### ⚠️ Accuracy Isn't Everything!

**Accuracy = (Correct Predictions) / (Total Predictions)**

```
Example:
Predicted:  [Pos, Neg, Pos, Pos, Neg]
Actual:     [Pos, Neg, Pos, Neg, Neg]
Accuracy:   4/5 = 80%
```

**But wait!** What if the model predicts everything as "Positive"?
```
Predicted:  [Pos, Pos, Pos, Pos, Pos]
Actual:     [Pos, Neg, Pos, Neg, Neg]
Accuracy:   3/5 = 60%

But we're wrong about negatives! This model is useless for finding negative reviews!
```

#### The 4 Key Metrics

**1. Precision** (of predicted positives, how many are actually positive?)
```
Predicted: [Pos, Pos, Pos, Neg, Neg]
Actual:    [Pos, Pos, Neg, Neg, Neg]
Precision = 2/3 ≈ 67%  (2 correct out of 3 positive predictions)
```
**When to care**: If false positives are expensive (wrong promotional material = lost money)

**2. Recall** (of actual positives, how many did we catch?)
```
Predicted: [Pos, Pos, Pos, Neg, Neg]
Actual:    [Pos, Pos, Neg, Pos, Neg]
Recall = 2/3 ≈ 67%  (2 caught out of 3 actual positives)
```
**When to care**: If false negatives are expensive (missed positive reviews = lost customers)

**3. F1 Score** (harmonic mean of precision & recall)
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
Used when you care about both false positives AND false negatives
```

**4. ROC-AUC** (Area Under the Receiver Operating Characteristic Curve)
```
- Measure of model's ability to distinguish between classes
- 0.5 = random guessing
- 1.0 = perfect classification
- Value of 0.95+ = excellent model
```

#### The Confusion Matrix
```
                Predicted Negative    Predicted Positive
Actual Negative        TN                  FP
Actual Positive        FN                  TP

TN = True Negative  (correctly predicted negative)
FP = False Positive (incorrectly predicted positive)
FN = False Negative (incorrectly predicted positive)
TP = True Positive  (correctly predicted positive)
```

---

### **Stage 7: Visualizations**

#### Why Visualize?
Numbers alone don't tell the full story. Visualizations reveal patterns.

**Chart 1: Model Comparison**
- Compare accuracy/ROC-AUC across all 5 models
- Identify the winner
- See which feature extraction (BoW vs TF-IDF) works better

**Chart 2: Confusion Matrix Heatmap**
- Visual representation of TP, TN, FP, FN
- Quickly spot if model struggles with one class

**Chart 3: ROC Curve**
- Shows performance at different decision thresholds
- Higher curve = better model
- Area under curve (AUC) = single number summary

**Chart 4: Top Predictive Words**
- Logistic Regression weights visualized
- See which words MOST strongly predict positive/negative
- Example: "masterpiece", "excellent" predict positive
- Example: "terrible", "awful" predict negative

**Chart 5: Word Clouds**
- Visual representation of word frequency
- Size = frequency (bigger words = appear more often)
- Positive reviews have words like: "excellent", "beautiful", "love"
- Negative reviews have words like: "bad", "waste", "terrible"

**Chart 6: Review Length Distribution**
- Are positive reviews longer than negative reviews?
- Helps understand if length affects sentiment classification

---

### **Stage 8: Production Pipeline & Live Inference**

#### What is a Pipeline?
A **Pipeline** bundles preprocessing and modeling into one object:

```python
Pipeline([
    ("tfidf", TfidfVectorizer(...)),  # Step 1: Convert text to numbers
    ("model", LogisticRegression(...)) # Step 2: Classify
])
```

**Why?** Prevents a common mistake:
```python
❌ WRONG:
tfidf.fit(X_train)         # Learn vocabulary from training data
tfidf.transform(X_test)    # Transform test data
tfidf.fit(X_test)          # ❌ MISTAKE! Learning from test data!

✅ RIGHT (with Pipeline):
pipeline.fit(X_train, y_train)     # Learn from training data only
pipeline.predict(X_test)           # Predict on test data
```

#### Live Inference Function
```python
def predict_sentiment(text: str) -> dict:
    processed = preprocess_text(text)
    proba = production_pipeline.predict_proba([processed])[0]
    return {
        "sentiment": "positive" or "negative",
        "confidence": percentage,
        "positive_prob": prob for positive,
        "negative_prob": prob for negative,
    }
```

**Example**:
```
Input:  "This movie is amazing!"
Output: {
    "sentiment": "positive",
    "confidence": "94.2%",
    "positive_prob": "94.2%",
    "negative_prob": "5.8%"
}
```

#### Testing
```python
Demo reviews:
1. "This is one of the greatest films..." → POSITIVE ✅
2. "Absolutely terrible. The acting was wooden..." → NEGATIVE ✅
3. "I don't think this film is good..." → NEGATIVE ✅
```

---

## Key Concepts Explained

### 1. **Natural Language Processing (NLP)**
The intersection of AI and human language. Techniques to make computers understand text.

### 2. **Text Classification**
Assigning labels to documents. Examples:
- Sentiment: positive/negative/neutral
- Spam detection: spam/not spam
- Topic classification: sports/politics/tech

### 3. **Tokenization**
Breaking text into words (tokens).
```
"I love movies" → ["I", "love", "movies"]
```

### 4. **Lemmatization vs Stemming**
Both reduce words to base form:
```
Lemmatization: "running" → "run" (using dictionary)
Stemming: "running" → "runn" (removing endings)

Lemmatization is more accurate but slower.
```

### 5. **Vectorization**
Converting words to numbers so ML models can use them.

### 6. **Overfitting vs Underfitting**
```
Underfitting:  Model too simple → poor performance on BOTH train & test
Overfitting:   Model memorized training data → excellent on train, poor on test
Goldilocks:    Model generalizes well → good on both train & test
```

### 7. **Train-Test Split**
```
Training set (80%): Model learns from this data
Test set (20%):    Evaluate model on unseen data
→ Prevents overfitting detection
```

### 8. **Cross-Validation**
Instead of one train-test split, use multiple:
```
Split 1: Train on 80%, test on 20%
Split 2: Train on different 80%, test on different 20%
Split 3: Repeat...
Average the results → more reliable evaluation
```

---

## How Everything Works Together

### The Complete Flow (Zoomed Out)

```
1. RAW DATA (50k .txt files)
   ↓
2. LOAD & COMBINE (Stage 0)
   → 50k reviews in one CSV
   ↓
3. EXPLORE (Stage 2)
   → Understand data distribution
   ↓
4. PREPROCESS (Stage 3)
   → Clean text (remove noise, lemmatize, etc.)
   ↓
5. VECTORIZE (Stage 4)
   → Convert text to TF-IDF vectors
   ↓
6. SPLIT (Stage 4)
   → 80% training, 20% testing
   ↓
7. TRAIN MODELS (Stage 5)
   → Fit 5 different models
   ↓
8. EVALUATE (Stage 6)
   → Check accuracy, precision, recall, AUC
   ↓
9. VISUALIZE (Stage 7)
   → Charts to understand results
   ↓
10. DEPLOY (Stage 8)
    → Pipeline ready for production
    → Predict sentiment on new reviews
```

### Key Insights

**Why does this work?**
1. **Data Quality**: 50,000 balanced, labeled reviews
2. **Good Preprocessing**: Removes noise while keeping signal
3. **Smart Vectorization**: TF-IDF weights important words
4. **Simple Model**: Logistic Regression learns clear patterns
5. **Proper Evaluation**: Multiple metrics avoid misleading accuracy

**Why use Logistic Regression?**
- Fast training
- Interpretable (see which words matter)
- Great performance (92%+ accuracy)
- Perfect for binary classification (positive/negative)

**Why not Deep Learning?**
- 50k reviews is enough for traditional ML
- No need for the complexity of neural networks
- Logistic Regression achieves 92%+ accuracy — great baseline
- (Deep learning would be overkill and slower)

---

## FastAPI + Joblib: Production Deployment

### ⚠️ The Problem with This Notebook

The model only lives in memory (RAM):
```python
# In the notebook:
production_pipeline.predict(["great movie"])  # ✅ Works
# Close the notebook...
# Open a new notebook...
production_pipeline  # ❌ NameError! It's gone!
```

### Solution: Save & Load with Joblib

**Joblib** is a library that **persists objects to disk**:

#### Step 1: Save the Model
```python
import joblib

joblib.dump(production_pipeline, 'sentiment_model.pkl')
print("✅ Model saved to sentiment_model.pkl")
```

This creates a file `sentiment_model.pkl` (pickle format) containing:
- The TF-IDF vectorizer (vocabulary + weights)
- The Logistic Regression model (word weights)

#### Step 2: Load the Model
```python
import joblib

model = joblib.load('sentiment_model.pkl')
print("✅ Model loaded!")

# Use it
prediction = model.predict(["I love this movie"])  # Works!
```

---

### FastAPI: REST API for Your Model

#### What is an API?
A web service that lets other programs talk to your model.

**Without API**: Only you in a Jupyter notebook can use it
**With API**: Anyone can send a request over the internet and get a prediction

#### Basic FastAPI Structure

```python
from fastapi import FastAPI
import joblib

app = FastAPI(title="Sentiment Analyzer")
model = joblib.load('sentiment_model.pkl')

@app.post("/predict")
async def predict(text: str):
    """
    Predict sentiment for a given text.
    
    Example:
    POST http://localhost:8000/predict?text=This%20movie%20is%20great
    """
    prediction = model.predict([text])[0]
    confidence = model.predict_proba([text])[0]
    
    return {
        "text": text,
        "sentiment": "positive" if prediction == 1 else "negative",
        "confidence": float(confidence.max()),
    }
```

#### Running the API
```bash
# Terminal:
pip install fastapi uvicorn

# Start server:
uvicorn app:app --reload --port 8000
```

Open browser: `http://localhost:8000/docs` → Interactive API documentation!

#### Using the API

**From Browser**:
```
http://localhost:8000/predict?text=This%20movie%20is%20great
```

**From Python**:
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    params={"text": "This movie is great"}
)
print(response.json())
# Output: {"sentiment": "positive", "confidence": 0.94}
```

**From JavaScript**:
```javascript
fetch("http://localhost:8000/predict?text=Great%20movie")
    .then(r => r.json())
    .then(data => console.log(data))
```

**From Command Line (curl)**:
```bash
curl "http://localhost:8000/predict?text=I%20love%20this"
```

---

### Complete Production Setup

#### Structure:
```
sentiment_api/
├── app.py                      # FastAPI application
├── sentiment_model.pkl         # Saved model (joblib)
├── preprocessing.py            # Preprocessing functions
├── requirements.txt            # Dependencies
└── README.md                   # Instructions
```

#### app.py
```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(
    title="IMDB Sentiment Analyzer",
    description="Predicts positive/negative sentiment on movie reviews",
    version="1.0"
)

# Load model once at startup (not on every request)
model = joblib.load('sentiment_model.pkl')

class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    positive_prob: float
    negative_prob: float

@app.get("/")
async def root():
    return {"message": "IMDB Sentiment API. Use /predict endpoint."}

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: TextInput):
    """
    Predict sentiment for movie review text.
    
    Args:
        text: Review text
        
    Returns:
        Sentiment prediction with confidence scores
    """
    probabilities = model.predict_proba([input_data.text])[0]
    prediction = np.argmax(probabilities)
    
    return PredictionResponse(
        sentiment="positive" if prediction == 1 else "negative",
        confidence=float(probabilities.max()),
        positive_prob=float(probabilities[1]),
        negative_prob=float(probabilities[0])
    )

@app.post("/batch_predict")
async def batch_predict(texts: list[str]):
    """Predict sentiment for multiple reviews at once."""
    results = []
    for text in texts:
        proba = model.predict_proba([text])[0]
        results.append({
            "text": text,
            "sentiment": "positive" if np.argmax(proba) == 1 else "negative",
            "confidence": float(proba.max())
        })
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### requirements.txt
```
fastapi==0.104.1
uvicorn==0.24.0
joblib==1.3.2
scikit-learn==1.3.2
pandas==2.1.3
numpy==1.26.2
```

#### Installation & Running
```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python app.py
# Or:
uvicorn app:app --reload --port 8000

# Visit: http://localhost:8000/docs
```

#### Interactive Documentation
FastAPI automatically generates **OpenAPI/Swagger documentation**:

```
GET  http://localhost:8000/docs
```

This gives you:
- List of all endpoints
- Input/output specifications
- "Try it out" button to test directly
- Example requests & responses

---

### Why This Setup?

**Advantages of Joblib + FastAPI**:
1. **Persistence**: Model survives server restarts
2. **Scalability**: API can handle multiple requests
3. **Language Agnostic**: Anyone can use your API (Python, JavaScript, mobile, etc.)
4. **Production Ready**: Proper error handling and documentation
5. **Easy Deployment**: Can run on cloud (AWS, Heroku, Azure, Google Cloud)

**Example Deployment Options**:
```
1. Local Machine: python app.py
2. Docker: docker run sentiment-api
3. Cloud (Heroku): git push heroku main
4. Cloud (AWS Lambda): serverless framework
5. Cloud (Google Cloud Run): gcloud run deploy
```

---

## Summary

### What You've Built
A **production-ready sentiment analysis system** that:
1. ✅ Loads 50k IMDB reviews
2. ✅ Preprocesses text intelligently
3. ✅ Extracts TF-IDF features
4. ✅ Trains 5 different models
5. ✅ Evaluates with multiple metrics
6. ✅ Visualizes results
7. ✅ Makes predictions on new text
8. ✅ Saves/loads with joblib
9. ✅ Exposes REST API with FastAPI

### Key Takeaways
- **Data Preprocessing** is 80% of the work
- **NLP** requires special techniques (lemmatization, stopwords, negations)
- **Vectorization** converts text to numbers
- **Simple models** (Logistic Regression) often work best
- **Multiple metrics** reveal true performance
- **Joblib + FastAPI** makes models production-ready

### Next Steps to Improve
1. **Hyperparameter tuning** with GridSearchCV
2. **Deep learning** (LSTM, BERT transformers)
3. **Cross-validation** for reliable evaluation
4. **Class weights** to handle imbalanced data
5. **Ensemble methods** combining multiple models
6. **Docker containerization** for easy deployment
7. **Monitoring** to track model performance in production

---

## Additional Resources

### Libraries Used
- **nltk**: Natural Language Processing
- **scikit-learn**: Machine Learning
- **pandas**: Data Analysis
- **matplotlib**: Visualization
- **fastapi**: REST API Framework
- **joblib**: Model Serialization

### Further Reading
- Sentiment Analysis Guide: https://huggingface.co/blog/sentiment-analysis
- NLTK Book: https://www.nltk.org/book/
- Scikit-learn Documentation: https://scikit-learn.org/
- FastAPI Tutorial: https://fastapi.tiangolo.com/tutorial/
- IMDB Dataset Paper: https://www.aclweb.org/anthology/P11-1015.pdf

---

**Created**: March 2026  
**Dataset**: IMDB aclImdb (50,000 reviews)  
**Best Model Accuracy**: ~92%+
