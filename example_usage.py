"""
🎯 PRACTICAL EXAMPLES: Using the Sentiment API
===============================================

This file shows real-world examples of how to use the FastAPI sentiment analyzer
in different scenarios.

Run the API first:
    python sentiment_api.py

Then try these examples!
"""

# ═════════════════════════════════════════════════════════════════════════════
# 📌 EXAMPLE 1: Basic Single Prediction
# ═════════════════════════════════════════════════════════════════════════════

def example_single_prediction():
    """Analyze sentiment of a single movie review."""
    import requests
    
    BASE_URL = "http://localhost:8000"
    
    # Single review to analyze
    review = "This movie is absolutely fantastic! Best film I've seen all year!"
    
    # Make request
    response = requests.post(
        f"{BASE_URL}/predict",
        json={"text": review}
    )
    
    # Handle response
    if response.status_code == 200:
        data = response.json()
        print(f"Review: {data['text']}")
        print(f"Sentiment: {data['sentiment'].upper()}")
        print(f"Confidence: {data['confidence']*100:.2f}%")
        print(f"  - Positive: {data['positive_prob']*100:.2f}%")
        print(f"  - Negative: {data['negative_prob']*100:.2f}%")
    else:
        print(f"Error: {response.status_code}")
        print(response.json())


# ═════════════════════════════════════════════════════════════════════════════
# 📌 EXAMPLE 2: Batch Analysis (Multiple Reviews)
# ═════════════════════════════════════════════════════════════════════════════

def example_batch_prediction():
    """Analyze sentiment for multiple reviews at once."""
    import requests
    
    BASE_URL = "http://localhost:8000"
    
    reviews = [
        "I absolutely loved every minute of this movie!",
        "Terrible film, complete waste of time.",
        "It was okay, nothing special really.",
        "Outstanding acting and brilliant cinematography!",
        "Don't bother watching this garbage.",
    ]
    
    # Batch request
    response = requests.post(
        f"{BASE_URL}/batch_predict",
        json={"texts": reviews}
    )
    
    if response.status_code == 200:
        data = response.json()
        
        print(f"Analyzed {data['total']} reviews in {data['processing_time_ms']:.2f}ms\n")
        
        for pred in data['predictions']:
            icon = "✅" if pred['sentiment'] == 'positive' else "❌"
            print(f"{icon} {pred['text'][:50]}")
            print(f"   → {pred['sentiment'].upper()} ({pred['confidence']*100:.1f}%)\n")


# ═════════════════════════════════════════════════════════════════════════════
# 📌 EXAMPLE 3: Real-World Use Case - Monitor Social Media Reviews
# ═════════════════════════════════════════════════════════════════════════════

def example_monitor_reviews():
    """
    Simulate monitoring customer reviews on a movie platform.
    Alert on negative reviews.
    """
    import requests
    
    BASE_URL = "http://localhost:8000"
    
    # Simulated reviews from a platform
    customer_reviews = [
        ("John", "This movie changed my life! Absolutely brilliant!"),
        ("Sarah", "Waste of money. Terrible plot and acting."),
        ("Mike", "Pretty entertaining, worth watching."),
        ("Lisa", "One of the worst films ever made. Don't watch!"),
        ("Tom", "Not bad, some good moments but overall average."),
    ]
    
    print("📊 MONITORING CUSTOMER REVIEWS\n")
    print("=" * 60)
    
    positive_count = 0
    negative_count = 0
    
    for username, review in customer_reviews:
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"text": review}
        )
        
        if response.status_code == 200:
            pred = response.json()
            sentiment = pred['sentiment']
            confidence = pred['confidence']
            
            if sentiment == 'positive':
                positive_count += 1
                status = "✅ POSITIVE"
            else:
                negative_count += 1
                status = "❌ NEGATIVE"
                # Alert: Negative review!
                print(f"\n🚨 ALERT: Negative review from {username}!")
                print(f"   Confidence: {confidence*100:.1f}%")
            
            print(f"\n{username}: {status} ({confidence*100:.1f}% confidence)")
            print(f"  Review: {review}")
    
    print("\n" + "=" * 60)
    print(f"\n📈 SUMMARY:")
    print(f"   Positive reviews: {positive_count}")
    print(f"   Negative reviews: {negative_count}")
    print(f"   Sentiment ratio: {positive_count/(positive_count+negative_count)*100:.1f}% positive")


# ═════════════════════════════════════════════════════════════════════════════
# 📌 EXAMPLE 4: Sentiment Trend Analysis
# ═════════════════════════════════════════════════════════════════════════════

def example_sentiment_trends():
    """
    Analyze how sentiment changes over time or across products.
    """
    import requests
    from collections import defaultdict
    
    BASE_URL = "http://localhost:8000"
    
    # Reviews grouped by movie
    movie_reviews = {
        "Movie A": [
            "Absolutely amazing!",
            "Fantastic film!",
            "Best movie ever!",
            "Not good at all.",
            "Terrible waste of time.",
        ],
        "Movie B": [
            "Very entertaining.",
            "Pretty good.",
            "Quite enjoyable.",
            "Not my cup of tea.",
            "Disappointing.",
        ],
        "Movie C": [
            "Perfect film!",
            "Stunning visuals!",
            "Brilliantly done!",
            "Absolutely horrible.",
            "Complete disaster.",
        ]
    }
    
    print("📈 SENTIMENT TREND ANALYSIS BY MOVIE\n")
    print("=" * 70)
    
    for movie_name, reviews in movie_reviews.items():
        response = requests.post(
            f"{BASE_URL}/batch_predict",
            json={"texts": reviews}
        )
        
        if response.status_code == 200:
            data = response.json()
            
            positive = sum(1 for p in data['predictions'] if p['sentiment'] == 'positive')
            negative = sum(1 for p in data['predictions'] if p['sentiment'] == 'negative')
            total = len(data['predictions'])
            
            positive_pct = (positive / total) * 100
            
            # Create simple bar chart
            bar_length = 30
            filled = int(bar_length * positive / total)
            bar = "█" * filled + "░" * (bar_length - filled)
            
            print(f"\n{movie_name}")
            print(f"  {bar} {positive_pct:.0f}% positive")
            print(f"  Positive: {positive}, Negative: {negative}")


# ═════════════════════════════════════════════════════════════════════════════
# 📌 EXAMPLE 5: Error Handling & Robustness
# ═════════════════════════════════════════════════════════════════════════════

def example_error_handling():
    """
    Properly handle errors and edge cases when using the API.
    """
    import requests
    
    BASE_URL = "http://localhost:8000"
    
    test_cases = [
        ("Valid review", "This movie is great!"),
        ("Empty string", ""),
        ("Very long text", "A" * 6000),  # Exceeds max_length
        ("Special characters", "!@#$%^&*()"),
        ("Numbers only", "1234567890"),
    ]
    
    print("🔍 ERROR HANDLING EXAMPLES\n")
    
    for test_name, text in test_cases:
        print(f"\nTest: {test_name}")
        print(f"Input: {text[:50]}..." if len(text) > 50 else f"Input: {text}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                json={"text": text},
                timeout=5  # 5-second timeout
            )
            
            if response.status_code == 200:
                pred = response.json()
                print(f"✅ Success: {pred['sentiment']} ({pred['confidence']*100:.1f}%)")
            
            elif response.status_code == 422:
                print(f"❌ Validation Error: Invalid input")
                errors = response.json()['detail']
                for error in errors:
                    print(f"   - {error['msg']}")
            
            elif response.status_code == 503:
                print(f"❌ Service Error: {response.json()['detail']}")
            
            else:
                print(f"❌ Error {response.status_code}: {response.json()}")
        
        except requests.exceptions.Timeout:
            print(f"❌ Timeout: Request took too long")
        
        except requests.exceptions.ConnectionError:
            print(f"❌ Connection Error: Cannot reach API")
        
        except Exception as e:
            print(f"❌ Unexpected error: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# 📌 EXAMPLE 6: Performance Benchmarking
# ═════════════════════════════════════════════════════════════════════════════

def example_performance_benchmark():
    """
    Measure API performance and latency.
    """
    import requests
    import time
    import statistics
    
    BASE_URL = "http://localhost:8000"
    
    test_review = "This is a great movie that I really enjoyed watching!"
    num_requests = 10
    
    print(f"⏱️  PERFORMANCE BENCHMARK ({num_requests} requests)\n")
    print("=" * 60)
    
    # Single prediction latency
    print("\n1️⃣  Single Predictions:")
    times_single = []
    
    for i in range(num_requests):
        start = time.time()
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"text": test_review}
        )
        elapsed = time.time() - start
        times_single.append(elapsed)
        print(f"  Request {i+1}: {elapsed*1000:.2f}ms")
    
    print(f"\n  Average: {statistics.mean(times_single)*1000:.2f}ms")
    print(f"  Min: {min(times_single)*1000:.2f}ms")
    print(f"  Max: {max(times_single)*1000:.2f}ms")
    print(f"  Stdev: {statistics.stdev(times_single)*1000:.2f}ms")
    
    # Batch prediction latency
    print("\n2️⃣  Batch Predictions (10 reviews):")
    reviews = [test_review] * 10
    
    start = time.time()
    response = requests.post(
        f"{BASE_URL}/batch_predict",
        json={"texts": reviews}
    )
    elapsed = time.time() - start
    
    print(f"  Time for 10 reviews: {elapsed*1000:.2f}ms")
    print(f"  Time per review: {elapsed*1000/10:.2f}ms")
    
    # Throughput
    print("\n3️⃣  Throughput:")
    print(f"  Single endpoint: {1/statistics.mean(times_single):.1f} requests/sec")
    print(f"  Batch endpoint: {10/elapsed:.1f} reviews/sec")


# ═════════════════════════════════════════════════════════════════════════════
# 📌 EXAMPLE 7: Web App Integration (Flask)
# ═════════════════════════════════════════════════════════════════════════════

"""
from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)
SENTIMENT_API = "http://localhost:8000"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    review = request.json.get("text")
    
    response = requests.post(
        f"{SENTIMENT_API}/predict",
        json={"text": review}
    )
    
    return jsonify(response.json())

if __name__ == "__main__":
    app.run(debug=True, port=5000)


# index.html
<html>
<head>
    <title>Movie Review Analyzer</title>
    <style>
        body { font-family: Arial; max-width: 600px; margin: 50px auto; }
        .sentiment-positive { color: green; font-weight: bold; }
        .sentiment-negative { color: red; font-weight: bold; }
    </style>
</head>
<body>
    <h1>Movie Review Sentiment Analyzer</h1>
    
    <textarea id="reviewText" rows="4" cols="50" 
              placeholder="Enter a movie review..."></textarea><br><br>
    
    <button onclick="analyzeSentiment()">Analyze</button>
    
    <div id="result" style="margin-top: 20px;"></div>
    
    <script>
        async function analyzeSentiment() {
            const text = document.getElementById('reviewText').value;
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: text})
            });
            const data = await response.json();
            
            const className = 'sentiment-' + data.sentiment;
            document.getElementById('result').innerHTML = `
                <p>Sentiment: <span class="${className}">${data.sentiment.toUpperCase()}</span></p>
                <p>Confidence: ${(data.confidence*100).toFixed(2)}%</p>
                <ul>
                    <li>Positive: ${(data.positive_prob*100).toFixed(2)}%</li>
                    <li>Negative: ${(data.negative_prob*100).toFixed(2)}%</li>
                </ul>
            `;
        }
    </script>
</body>
</html>
"""


# ═════════════════════════════════════════════════════════════════════════════
# 📌 EXAMPLE 8: Command-Line Tool
# ═════════════════════════════════════════════════════════════════════════════

def example_cli_tool():
    """
    Interactive command-line tool for sentiment analysis.
    """
    import requests
    
    BASE_URL = "http://localhost:8000"
    
    print("🎬 SENTIMENT ANALYZER CLI")
    print("=" * 60)
    print("Commands:")
    print("  'predict <text>'     - Analyze a single review")
    print("  'batch'              - Analyze multiple reviews")
    print("  'health'             - Check API status")
    print("  'quit'               - Exit")
    print("=" * 60)
    
    while True:
        command = input("\n> ").strip()
        
        if command.startswith("predict "):
            review = command[8:]
            response = requests.post(
                f"{BASE_URL}/predict",
                json={"text": review}
            )
            if response.status_code == 200:
                data = response.json()
                print(f"Sentiment: {data['sentiment'].upper()}")
                print(f"Confidence: {data['confidence']*100:.2f}%")
            else:
                print(f"Error: {response.json()}")
        
        elif command == "batch":
            reviews = []
            print("Enter reviews (empty line to finish):")
            while True:
                line = input("  > ").strip()
                if not line:
                    break
                reviews.append(line)
            
            response = requests.post(
                f"{BASE_URL}/batch_predict",
                json={"texts": reviews}
            )
            
            if response.status_code == 200:
                data = response.json()
                for pred in data['predictions']:
                    print(f"  {pred['sentiment']}: {pred['text'][:40]}")
            else:
                print(f"Error: {response.json()}")
        
        elif command == "health":
            response = requests.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"Status: {data['status']}")
                print(f"Model: {'✅ Loaded' if data['model_loaded'] else '❌ Not loaded'}")
            else:
                print("API not responding")
        
        elif command == "quit":
            break
        
        else:
            print("Unknown command. Try 'predict', 'batch', 'health', or 'quit'")


# ═════════════════════════════════════════════════════════════════════════════
# 🚀 RUN EXAMPLES
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    examples = {
        "1": ("Single Prediction", example_single_prediction),
        "2": ("Batch Prediction", example_batch_prediction),
        "3": ("Monitor Reviews", example_monitor_reviews),
        "4": ("Sentiment Trends", example_sentiment_trends),
        "5": ("Error Handling", example_error_handling),
        "6": ("Performance Benchmark", example_performance_benchmark),
        "8": ("CLI Tool", example_cli_tool),
    }
    
    print("\n" + "=" * 60)
    print("🎯 SENTIMENT API - PRACTICAL EXAMPLES")
    print("=" * 60)
    print("\nMake sure the API is running: python sentiment_api.py\n")
    print("Available examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    print("  0. Run all examples")
    print("  q. Quit\n")
    
    choice = input("Select example (0-8): ").strip()
    
    if choice == "0":
        for key, (name, func) in examples.items():
            print(f"\n{'=' * 60}")
            print(f"Running: {name}")
            print(f"{'=' * 60}")
            try:
                func()
            except Exception as e:
                print(f"Error: {e}")
    
    elif choice in examples:
        name, func = examples[choice]
        print(f"\nRunning: {name}\n")
        try:
            func()
        except Exception as e:
            print(f"Error: {e}")
            print("\nMake sure the API is running: python sentiment_api.py")
    
    else:
        print("Invalid choice")
