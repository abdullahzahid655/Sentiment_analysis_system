# 🎬 IMDB Sentiment Analysis Project

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange?style=flat&logo=scikit-learn)
![NLTK](https://img.shields.io/badge/NLTK-3.8.1-green?style=flat)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.0-red?style=flat&logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)

---

## 📊 Project Overview

A complete, production-ready **Sentiment Analysis pipeline** trained on the IMDB movie reviews dataset. This project demonstrates end-to-end Natural Language Processing (NLP) with **92.1% accuracy** and deploys a REST API for real-time predictions.

| Metric | Value |
|--------|-------|
| **Dataset Size** | 50,000 reviews |
| **Model Accuracy** | 92.1% |
| **ROC-AUC Score** | 0.97 |
| **Inference Time** | ~10ms per review |

---

## 🔬 Results & Visualizations

### 📈 Rating Distribution
![Rating Distribution](outputs/00_rating_distribution.png)

### 🏆 Model Comparison
![Model Comparison](outputs/01_model_comparison.png)

### 📊 Confusion Matrix
![Confusion Matrix](outputs/02_confusion_matrix.png)

### 📉 ROC Curve
![ROC Curve](outputs/03_roc_curve.png)

### 🔤 Top Words by Sentiment
![Top Words](outputs/04_top_words.png)

### ☁️ Word Clouds
![Word Clouds](outputs/05_wordclouds.png)

### 📏 Review Length Distribution
![Length Distribution](outputs/06_length_distribution.png)

---

## 🏗️ Project Architecture

```
NLP_course/task_1/
├── sentiment_analysis_nlp.ipynb   # Main notebook (8 stages)
├── sentiment_api.py               # FastAPI REST API
├── example_usage.py                # Usage examples
├── requirements.txt                # Dependencies
├── README.md                       # This file
├── outputs/                        # Visualizations
│   ├── 00_rating_distribution.png
│   ├── 01_model_comparison.png
│   ├── 02_confusion_matrix.png
│   ├── 03_roc_curve.png
│   ├── 04_top_words.png
│   ├── 05_wordclouds.png
│   └── 06_length_distribution.png
├── aclImdb/                        # Original dataset
│   ├── train/                      # 25,000 training reviews
│   └── test/                       # 25,000 test reviews
└── joblib/                         # Saved models
```

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone <your-repo-url>
cd NLP_course/task_1
pip install -r requirements.txt
```

### 2. Run the Notebook
```bash
jupyter notebook sentiment_analysis_nlp.ipynb
```

### 3. Start the API
```bash
python sentiment_api.py
```

### 4. Test the API
Visit: `http://localhost:8000/docs`

---

## 💡 Key Features

- ✅ **7-Step Text Preprocessing Pipeline** - Cleaning, tokenization, stopwords removal, lemmatization
- ✅ **Multiple Feature Extraction Methods** - Bag of Words (BoW) & TF-IDF
- ✅ **5 Machine Learning Models** - Logistic Regression, Naive Bayes, SVM, Random Forest, SGD
- ✅ **Comprehensive Evaluation** - Accuracy, Precision, Recall, F1-Score, ROC-AUC
- ✅ **Production-Ready API** - FastAPI with request validation, error handling, CORS
- ✅ **Interactive Documentation** - Auto-generated API docs with Swagger UI


---

## 🔧 Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-FF9F1C?style=for-the-badge&logo=matplotlib&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3366CC?style=for-the-badge&logo=nltk&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)

</div>

---

## 📝 License

MIT License - Feel free to use this project for learning or commercial purposes!

---

## 👤 Author

**Your Name**
- 🌐 GitHub: https://github.com/abdullahzahid655
- 💼 LinkedIn: https://www.linkedin.com/in/abdullahzahid655

---

## 🙏 Acknowledgments

- [IMDB Dataset](http://ai.stanford.edu/~amaas/data/sentiment/) - Stanford AI Lab
- [Elevvo](https://linkedin.com/company/elevvo) - For the learning opportunity

---

<div align="center">

**⭐ Star this repo if you found it helpful!**

*Built with ❤️ using Python & Machine Learning*

</div>

