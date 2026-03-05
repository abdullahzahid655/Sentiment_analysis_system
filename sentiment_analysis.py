"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          SENTIMENT ANALYSIS ON PRODUCT REVIEWS — NLP Task 1                ║
║          Professional Industrial-Grade Implementation                        ║
║          Tools: Python | Pandas | NLTK | Scikit-learn                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

LEARNING ROADMAP:
  STAGE 1 → Setup & Imports
  STAGE 2 → Dataset Loading & Exploration
  STAGE 3 → Text Preprocessing Pipeline
  STAGE 4 → Feature Extraction (CountVectorizer + TF-IDF)
  STAGE 5 → Model Training (Logistic Regression + Naive Bayes)
  STAGE 6 → Evaluation & Metrics
  STAGE 7 → Visualization (Word Clouds + Frequency Plots)
  STAGE 8 → Prediction on New Text (Inference)
"""

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1: IMPORTS & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# Why: Every professional NLP project starts with organized imports and config.
#      This makes the codebase readable and maintainable.

import os
import re
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter

# --- NLP Libraries ---
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

# --- Scikit-learn: Feature Extraction ---
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# --- Scikit-learn: Models ---
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB

# --- Scikit-learn: Evaluation ---
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline

# --- Word Cloud (Bonus) ---
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    print("⚠ WordCloud not installed. Run: pip install wordcloud")

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
RANDOM_STATE   = 42          # For reproducibility
TEST_SIZE      = 0.2         # 80% train, 20% test
MAX_FEATURES   = 10_000      # Vocabulary size cap for vectorizers
NGRAM_RANGE    = (1, 2)      # Unigrams + Bigrams (captures "not good", "very bad")
OUTPUT_DIR     = "outputs"   # Directory to save plots & reports

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Download required NLTK data ───────────────────────────────────────────────
print("─" * 65)
print("  STAGE 1: Downloading NLTK resources...")
print("─" * 65)

for resource in ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger", "punkt_tab"]:
    nltk.download(resource, quiet=True)

print("  ✓ NLTK resources ready\n")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2: DATASET LOADING & EXPLORATION
# ─────────────────────────────────────────────────────────────────────────────
# WHY: Understanding your data is the most critical step in any ML project.
#      We use the IMDB dataset from Hugging Face Datasets (or a fallback
#      synthetic dataset if offline), then explore its shape, balance,
#      and text characteristics.

print("─" * 65)
print("  STAGE 2: Loading Dataset & Exploratory Data Analysis")
print("─" * 65)

def load_dataset() -> pd.DataFrame:
    """
    Load IMDB sentiment dataset.
    Priority:
      1. Hugging Face `datasets` library (real IMDB — 50k reviews)
      2. Synthetic fallback (for offline environments)
    """
    try:
        from datasets import load_dataset as hf_load
        print("  → Loading IMDB dataset from Hugging Face (50,000 reviews)...")
        ds = hf_load("imdb")
        # Combine train + test splits for our own split later
        train_df = pd.DataFrame(ds["train"]).rename(columns={"label": "sentiment"})
        test_df  = pd.DataFrame(ds["test"]).rename(columns={"label": "sentiment"})
        df = pd.concat([train_df, test_df], ignore_index=True)
        df["sentiment"] = df["sentiment"].map({0: "negative", 1: "positive"})
        print(f"  ✓ Loaded {len(df):,} reviews from IMDB\n")
        return df

    except Exception:
        print("  ⚠ Hugging Face not available. Generating synthetic dataset...")
        return _generate_synthetic_dataset(n_samples=2000)


def _generate_synthetic_dataset(n_samples: int = 2000) -> pd.DataFrame:
    """
    Create a balanced synthetic sentiment dataset for offline learning.
    Each review is assembled from positive/negative phrase pools to simulate
    real-world variation.
    """
    np.random.seed(RANDOM_STATE)
    pos_phrases = [
        "absolutely loved this product", "excellent quality and fast shipping",
        "highly recommend to everyone", "works perfectly out of the box",
        "amazing value for the money", "exceeded my expectations completely",
        "best purchase I have made this year", "fantastic build quality",
        "incredible performance and reliability", "five stars without hesitation",
        "very satisfied with this item", "product arrived quickly and in great condition",
    ]
    neg_phrases = [
        "total waste of money", "broke after just one week",
        "extremely disappointed with quality", "does not work as advertised",
        "terrible customer service experience", "would not recommend to anyone",
        "cheaply made and overpriced", "arrived damaged and unusable",
        "stopped working after a few days", "complete garbage product",
        "very poor build quality", "this is the worst product I have ever bought",
    ]
    reviews, labels = [], []
    for _ in range(n_samples // 2):
        review = ". ".join(np.random.choice(pos_phrases, np.random.randint(2, 5), replace=True))
        reviews.append(review); labels.append("positive")
    for _ in range(n_samples // 2):
        review = ". ".join(np.random.choice(neg_phrases, np.random.randint(2, 5), replace=True))
        reviews.append(review); labels.append("negative")

    df = pd.DataFrame({"text": reviews, "sentiment": labels}).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"  ✓ Generated {len(df):,} synthetic reviews\n")
    return df


def explore_dataset(df: pd.DataFrame) -> None:
    """Print key statistics and class distribution for the dataset."""
    print("  📊 DATASET OVERVIEW")
    print(f"     Total reviews    : {len(df):,}")
    print(f"     Positive reviews : {(df['sentiment']=='positive').sum():,} ({(df['sentiment']=='positive').mean()*100:.1f}%)")
    print(f"     Negative reviews : {(df['sentiment']=='negative').sum():,} ({(df['sentiment']=='negative').mean()*100:.1f}%)")
    df["review_length"] = df["text"].apply(lambda x: len(x.split()))
    print(f"\n  📝 REVIEW LENGTH STATS (words)")
    print(f"     Mean   : {df['review_length'].mean():.0f}")
    print(f"     Median : {df['review_length'].median():.0f}")
    print(f"     Min    : {df['review_length'].min()}")
    print(f"     Max    : {df['review_length'].max():,}")
    print()


# Load & explore
df = load_dataset()
explore_dataset(df)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3: TEXT PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
# WHY: Raw text is messy — HTML tags, punctuation, numbers, uppercase letters,
#      and filler "stopwords" (the, is, at, ...) add noise and inflate the
#      vocabulary without contributing meaning.
#      Preprocessing standardizes text so models can learn real patterns.

print("─" * 65)
print("  STAGE 3: Text Preprocessing Pipeline")
print("─" * 65)

# Initialize NLP tools
lemmatizer  = WordNetLemmatizer()
stemmer     = PorterStemmer()
STOP_WORDS  = set(stopwords.words("english"))

# Keep negation words — they flip sentiment! ("not good" ≠ "good")
NEGATION_WORDS = {"no", "not", "nor", "never", "neither", "nobody", "nothing",
                  "nowhere", "hardly", "barely", "scarcely", "without"}
STOP_WORDS -= NEGATION_WORDS   # Remove negation words from stopword list


def clean_text(text: str) -> str:
    """
    Step-by-step text cleaning:
      1. Lowercase
      2. Remove HTML tags (important for web-scraped reviews)
      3. Expand contractions (don't → do not, can't → can not)
      4. Remove special characters / numbers
      5. Remove extra whitespace
    """
    # 1. Lowercase
    text = text.lower()

    # 2. Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # 3. Expand common contractions
    contractions = {
        "n't": " not", "'re": " are", "'s": " is", "'d": " would",
        "'ll": " will", "'t": " not", "'ve": " have", "'m": " am",
        "can't": "can not", "won't": "will not", "don't": "do not",
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)

    # 4. Keep only alphabetic characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # 5. Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_text(text: str, use_lemmatization: bool = True) -> str:
    """
    Full NLP preprocessing pipeline:
      clean → tokenize → remove stopwords → lemmatize/stem → rejoin
    
    Args:
        text: Raw review string
        use_lemmatization: True = lemmatize (better), False = stem (faster)
    
    Returns:
        Processed text as a single string
    """
    # Step 1: Clean
    text = clean_text(text)

    # Step 2: Tokenize (split into list of words)
    tokens = word_tokenize(text)

    # Step 3: Remove stopwords + short tokens (< 2 chars are noise)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    # Step 4: Normalize words
    if use_lemmatization:
        # Lemmatization: "running" → "run", "better" → "good" (uses vocabulary)
        tokens = [lemmatizer.lemmatize(t, pos="v") for t in tokens]
        tokens = [lemmatizer.lemmatize(t, pos="n") for t in tokens]
    else:
        # Stemming: "running" → "run" (crude but faster, removes suffixes)
        tokens = [stemmer.stem(t) for t in tokens]

    # Step 5: Rejoin tokens back into a string (required by vectorizers)
    return " ".join(tokens)


# ── Apply preprocessing ───────────────────────────────────────────────────────
print("  → Preprocessing all reviews (this may take a minute)...")
start = time.time()
df["processed_text"] = df["text"].apply(preprocess_text)
elapsed = time.time() - start
print(f"  ✓ Preprocessing complete in {elapsed:.1f}s\n")

# Show a before/after example
sample_idx = df[df["sentiment"] == "negative"].index[0]
print("  📋 PREPROCESSING EXAMPLE:")
print(f"     Original : {df.loc[sample_idx, 'text'][:120]}...")
print(f"     Processed: {df.loc[sample_idx, 'processed_text'][:120]}...")
print()


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4: FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
# WHY: ML models work with numbers, not text.
#      We must convert our words into numerical vectors.
#
#   METHOD 1 — CountVectorizer (Bag of Words):
#     Each document becomes a vector of word counts.
#     Simple and interpretable. Ignores order.
#     Example: "good movie" → [0,0,1,0,1,0,...] (sparse vector)
#
#   METHOD 2 — TF-IDF (Term Frequency × Inverse Document Frequency):
#     Weights words by how often they appear in a document vs. the corpus.
#     Rare but meaningful words get HIGH scores.
#     Common words ("the", "is") get LOW scores even if frequent.
#     TF-IDF is almost always better than raw counts.

print("─" * 65)
print("  STAGE 4: Feature Extraction (BoW + TF-IDF)")
print("─" * 65)

# ── Encode labels ─────────────────────────────────────────────────────────────
df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})

# ── Train / Test Split ────────────────────────────────────────────────────────
# stratify=y ensures both splits maintain the same positive/negative ratio
X = df["processed_text"].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"  ✓ Training set : {len(X_train):,} samples")
print(f"  ✓ Test set     : {len(X_test):,} samples\n")

# ── CountVectorizer ───────────────────────────────────────────────────────────
bow_vectorizer = CountVectorizer(
    max_features=MAX_FEATURES,
    ngram_range=NGRAM_RANGE,    # (1,2) = unigrams + bigrams
    min_df=2,                   # ignore terms appearing in fewer than 2 docs
    max_df=0.95,                # ignore terms in more than 95% of documents
)

X_train_bow = bow_vectorizer.fit_transform(X_train)   # LEARN vocabulary on train
X_test_bow  = bow_vectorizer.transform(X_test)        # APPLY vocabulary to test

# ── TF-IDF Vectorizer ─────────────────────────────────────────────────────────
tfidf_vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    ngram_range=NGRAM_RANGE,
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,    # Apply log(1+tf) smoothing — reduces impact of very frequent terms
)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf  = tfidf_vectorizer.transform(X_test)

print(f"  ✓ BoW  matrix shape : {X_train_bow.shape}  (samples × vocab)")
print(f"  ✓ TF-IDF matrix shape: {X_train_tfidf.shape}")
print(f"  ✓ Vocabulary size    : {len(tfidf_vectorizer.vocabulary_):,} terms\n")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5: MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
# WHY: We train multiple classifiers so we can compare their strengths.
#
#   LOGISTIC REGRESSION:
#     Despite the name, it's a LINEAR CLASSIFIER.
#     It learns a weight for each word/feature. Positive weight → positive
#     sentiment. Negative weight → negative sentiment.
#     Very interpretable. Often the best choice for text classification.
#
#   NAIVE BAYES (MultinomialNB):
#     A probabilistic classifier based on Bayes' theorem.
#     Assumes features are INDEPENDENT (naive assumption — usually wrong
#     but still works well in practice for text).
#     Very fast, good baseline, handles sparse data well.
#
#   COMPLEMENT NAIVE BAYES:
#     An improved variant of MultinomialNB that's especially good for
#     imbalanced datasets.

print("─" * 65)
print("  STAGE 5: Model Training")
print("─" * 65)

# ── Define all model experiments ─────────────────────────────────────────────
experiments = {
    "LogReg + TF-IDF" : (LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE), X_train_tfidf, X_test_tfidf),
    "LogReg + BoW"    : (LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE), X_train_bow,   X_test_bow),
    "NaiveBayes + TF-IDF": (MultinomialNB(alpha=0.1),                                         X_train_tfidf, X_test_tfidf),
    "NaiveBayes + BoW"   : (MultinomialNB(alpha=0.1),                                         X_train_bow,   X_test_bow),
    "ComplementNB + TF-IDF": (ComplementNB(alpha=0.1),                                        X_train_tfidf, X_test_tfidf),
}

results   = {}
models    = {}

for name, (model, X_tr, X_te) in experiments.items():
    t0 = time.time()
    model.fit(X_tr, y_train)
    train_time = time.time() - t0

    y_pred     = model.predict(X_te)
    y_prob     = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0

    results[name] = {"accuracy": acc, "auc": auc, "train_time": train_time}
    models[name]  = (model, X_tr, X_te, y_pred, y_prob)

    print(f"  ✓ [{name}]")
    print(f"     Accuracy: {acc*100:.2f}%  |  ROC-AUC: {auc:.4f}  |  Train time: {train_time:.2f}s")

print()


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6: EVALUATION & METRICS
# ─────────────────────────────────────────────────────────────────────────────
# WHY: Accuracy alone is misleading. Professional evaluation uses:
#   • Precision  — of all predicted positives, how many are truly positive?
#   • Recall     — of all actual positives, how many did we catch?
#   • F1-Score   — harmonic mean of precision and recall
#   • ROC-AUC    — measures model's ability to discriminate between classes
#   • Confusion Matrix — shows exactly where the model makes mistakes

print("─" * 65)
print("  STAGE 6: Detailed Evaluation")
print("─" * 65)

# ── Best model ────────────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["accuracy"])
best_model, _, _, best_preds, best_probs = models[best_name]
print(f"\n  🏆 BEST MODEL: {best_name}")
print(f"     Accuracy : {results[best_name]['accuracy']*100:.2f}%")
print(f"     ROC-AUC  : {results[best_name]['auc']:.4f}")
print()
print("  📋 FULL CLASSIFICATION REPORT:")
print(classification_report(y_test, best_preds, target_names=["Negative", "Positive"]))


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 7: VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────
print("─" * 65)
print("  STAGE 7: Generating Visualizations...")
print("─" * 65)

# ── Style setup ───────────────────────────────────────────────────────────────
plt.style.use("dark_background")
PALETTE_POS = "#2ecc71"   # green
PALETTE_NEG = "#e74c3c"   # red
PALETTE_ACC = "#3498db"   # blue
BG_COLOR    = "#0d1117"
CARD_COLOR  = "#161b22"


def save_fig(filename: str) -> None:
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"  ✓ Saved → {path}")


# ── Plot 1: Model Comparison ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG_COLOR)
fig.suptitle("Model Comparison Dashboard", color="white", fontsize=16, fontweight="bold", y=1.02)

model_names = list(results.keys())
accuracies  = [results[m]["accuracy"] * 100 for m in model_names]
aucs        = [results[m]["auc"] for m in model_names]
colors      = [PALETTE_POS if m == best_name else PALETTE_ACC for m in model_names]

# Accuracy bar chart
ax = axes[0]
ax.set_facecolor(CARD_COLOR)
bars = ax.barh(model_names, accuracies, color=colors, edgecolor="none", height=0.5)
ax.set_xlabel("Accuracy (%)", color="white")
ax.set_title("Accuracy by Model", color="white", fontweight="bold")
ax.tick_params(colors="white")
for bar, val in zip(bars, accuracies):
    ax.text(val - 0.5, bar.get_y() + bar.get_height() / 2, f"{val:.2f}%",
            va="center", ha="right", color="white", fontsize=9, fontweight="bold")

# ROC-AUC bar chart
ax = axes[1]
ax.set_facecolor(CARD_COLOR)
bars = ax.barh(model_names, aucs, color=colors, edgecolor="none", height=0.5)
ax.set_xlabel("ROC-AUC Score", color="white")
ax.set_title("ROC-AUC by Model", color="white", fontweight="bold")
ax.tick_params(colors="white")
for bar, val in zip(bars, aucs):
    ax.text(val - 0.002, bar.get_y() + bar.get_height() / 2, f"{val:.4f}",
            va="center", ha="right", color="white", fontsize=9, fontweight="bold")

plt.tight_layout()
save_fig("01_model_comparison.png")


# ── Plot 2: Confusion Matrix ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6), facecolor=BG_COLOR)
ax.set_facecolor(CARD_COLOR)
cm = confusion_matrix(y_test, best_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"Confusion Matrix — {best_name}", color="white", fontweight="bold")
ax.tick_params(colors="white")
ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")
plt.tight_layout()
save_fig("02_confusion_matrix.png")


# ── Plot 3: ROC Curve ─────────────────────────────────────────────────────────
if best_probs is not None:
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_COLOR)
    fpr, tpr, _ = roc_curve(y_test, best_probs)
    ax.plot(fpr, tpr, color=PALETTE_POS, lw=2, label=f"ROC (AUC = {results[best_name]['auc']:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Classifier")
    ax.set_xlabel("False Positive Rate", color="white")
    ax.set_ylabel("True Positive Rate", color="white")
    ax.set_title(f"ROC Curve — {best_name}", color="white", fontweight="bold")
    ax.tick_params(colors="white")
    ax.legend(loc="lower right", facecolor=CARD_COLOR, labelcolor="white")
    plt.tight_layout()
    save_fig("03_roc_curve.png")


# ── Plot 4: Top Predictive Words ──────────────────────────────────────────────
def get_top_features(model, vectorizer, n: int = 20):
    """Extract top positive and negative words from a logistic regression model."""
    feature_names = vectorizer.get_feature_names_out()
    coef = model.coef_[0]
    top_pos_idx = coef.argsort()[-n:][::-1]
    top_neg_idx = coef.argsort()[:n]
    return (
        [(feature_names[i], coef[i]) for i in top_pos_idx],
        [(feature_names[i], coef[i]) for i in top_neg_idx],
    )

# Use best LogReg model for interpretability
lr_model_name = "LogReg + TF-IDF"
lr_model = models[lr_model_name][0]

pos_words, neg_words = get_top_features(lr_model, tfidf_vectorizer, n=20)

fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG_COLOR)
fig.suptitle("Top 20 Most Predictive Words (Logistic Regression Weights)",
             color="white", fontsize=14, fontweight="bold")

for ax, word_scores, color, title in [
    (axes[0], pos_words, PALETTE_POS, "Strongest POSITIVE Indicators"),
    (axes[1], neg_words, PALETTE_NEG, "Strongest NEGATIVE Indicators"),
]:
    ax.set_facecolor(CARD_COLOR)
    words  = [w for w, _ in word_scores]
    scores = [abs(s) for _, s in word_scores]
    y_pos  = range(len(words))
    ax.barh(y_pos, scores, align="center", color=color, alpha=0.85, edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words, color="white")
    ax.set_xlabel("Coefficient Magnitude", color="white")
    ax.set_title(title, color="white", fontweight="bold")
    ax.tick_params(colors="white")

plt.tight_layout()
save_fig("04_top_words.png")


# ── Plot 5: Word Cloud (Bonus) ────────────────────────────────────────────────
if WORDCLOUD_AVAILABLE:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG_COLOR)
    fig.suptitle("Word Clouds by Sentiment", color="white", fontsize=14, fontweight="bold")

    pos_text = " ".join(df[df["sentiment"] == "positive"]["processed_text"])
    neg_text = " ".join(df[df["sentiment"] == "negative"]["processed_text"])

    for ax, text, colormap, title in [
        (axes[0], pos_text, "Greens",  "Positive Reviews"),
        (axes[1], neg_text, "Reds",    "Negative Reviews"),
    ]:
        wc = WordCloud(
            width=800, height=500, background_color="#0d1117",
            colormap=colormap, max_words=150, prefer_horizontal=0.9,
            stopwords=STOP_WORDS,
        ).generate(text)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, color="white", fontsize=13, fontweight="bold")

    plt.tight_layout()
    save_fig("05_wordclouds.png")
else:
    print("  ⚠ WordCloud skipped (not installed). Run: pip install wordcloud")


# ── Plot 6: Review Length Distribution ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG_COLOR)
ax.set_facecolor(CARD_COLOR)

for sentiment, color, label in [("positive", PALETTE_POS, "Positive"), ("negative", PALETTE_NEG, "Negative")]:
    lengths = df[df["sentiment"] == sentiment]["review_length"] if "review_length" in df.columns \
              else df[df["sentiment"] == sentiment]["text"].apply(lambda x: len(x.split()))
    ax.hist(lengths.clip(upper=500), bins=50, alpha=0.6, color=color, label=label, edgecolor="none")

ax.set_xlabel("Review Length (words)", color="white")
ax.set_ylabel("Count", color="white")
ax.set_title("Distribution of Review Lengths by Sentiment", color="white", fontweight="bold")
ax.tick_params(colors="white")
ax.legend(facecolor=CARD_COLOR, labelcolor="white")
plt.tight_layout()
save_fig("06_length_distribution.png")

print()


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 8: SKLEARN PIPELINE — PRODUCTION-READY INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
# WHY: In real production systems, you bundle preprocessing + model into a
#      PIPELINE. This ensures:
#        - No data leakage (vectorizer fitted only on training data)
#        - Single callable object for deployment
#        - Reproducible inference on new data

print("─" * 65)
print("  STAGE 8: Production Pipeline & Inference Demo")
print("─" * 65)

# Build a clean pipeline
production_pipeline = Pipeline([
    ("tfidf",  TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE,
                               min_df=2, max_df=0.95, sublinear_tf=True)),
    ("model",  LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)),
])

# Train on preprocessed text
X_train_processed = [preprocess_text(t) for t in df["text"].iloc[:int(len(df) * 0.8)]]
y_train_all       = df["label"].iloc[:int(len(df) * 0.8)].values
production_pipeline.fit(X_train_processed, y_train_all)


def predict_sentiment(text: str) -> dict:
    """
    Run full inference on raw text.
    Returns sentiment label + confidence score.
    
    Args:
        text: Raw review string (as a user would type it)
    
    Returns:
        dict with keys: sentiment, confidence, processed_text
    """
    processed = preprocess_text(text)
    proba     = production_pipeline.predict_proba([processed])[0]
    label_idx = proba.argmax()
    label     = "positive" if label_idx == 1 else "negative"
    confidence = proba[label_idx]

    return {
        "sentiment"     : label,
        "confidence"    : f"{confidence*100:.1f}%",
        "positive_prob" : f"{proba[1]*100:.1f}%",
        "negative_prob" : f"{proba[0]*100:.1f}%",
        "processed_text": processed,
    }


# ── Demo predictions ──────────────────────────────────────────────────────────
demo_reviews = [
    "This product is absolutely fantastic! Best purchase I've made all year. Highly recommend!",
    "Total garbage. Broke after two days. Completely waste of money. Stay away from this brand.",
    "It's okay, nothing special. Works as expected but quality could be better for the price.",
    "Not bad at all, actually quite impressed with the packaging and fast delivery.",
    "I wouldn't recommend this to anyone. Terrible customer service and the item was damaged.",
]

print("  🔍 LIVE INFERENCE DEMO:\n")
for review in demo_reviews:
    result = predict_sentiment(review)
    icon   = "✅" if result["sentiment"] == "positive" else "❌"
    print(f"  {icon} Review   : {review[:75]}...")
    print(f"     Sentiment : {result['sentiment'].upper()} ({result['confidence']} confidence)")
    print(f"     Pos/Neg   : {result['positive_prob']} / {result['negative_prob']}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("═" * 65)
print("  📊 FINAL RESULTS SUMMARY")
print("═" * 65)
print(f"  {'Model':<30} {'Accuracy':>10} {'ROC-AUC':>10}")
print("  " + "─" * 55)
for name in sorted(results, key=lambda k: results[k]["accuracy"], reverse=True):
    marker = " ← BEST" if name == best_name else ""
    print(f"  {name:<30} {results[name]['accuracy']*100:>9.2f}% {results[name]['auc']:>10.4f}{marker}")
print()
print(f"  📁 All plots saved to: ./{OUTPUT_DIR}/")
print("═" * 65)
print()
print("  NEXT STEPS TO IMPROVE:")
print("  1. Hyperparameter tuning with GridSearchCV")
print("  2. Try deep learning: LSTM or BERT (transformers library)")
print("  3. Handle class imbalance with SMOTE or class_weight='balanced'")
print("  4. Cross-validation for more reliable evaluation")
print("  5. Deploy as REST API using FastAPI + pickle/joblib")
print("═" * 65)
