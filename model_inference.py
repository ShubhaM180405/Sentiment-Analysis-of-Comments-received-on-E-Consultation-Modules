# model_inference.py
from transformers import pipeline

# Load pre-trained BERT model for sentiment analysis
sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiment(comment: str):
    """
    Analyze sentiment of a single comment using pre-trained BERT model.
    Returns: dict with label and score
    """
    result = sentiment_pipeline(comment)[0]
    return {
        "comment": comment,
        "label": result['label'],   # POSITIVE / NEGATIVE
        "score": round(result['score'], 3)
    }

def analyze_batch(comments: list):
    """
    Analyze sentiment of a list of comments.
    Returns: list of dicts
    """
    results = sentiment_pipeline(comments)
    output = []
    for comment, res in zip(comments, results):
        output.append({
            "comment": comment,
            "label": res['label'],
            "score": round(res['score'], 3)
        })
    return output

if __name__ == "__main__":
    # Quick test
    test_comments = [
        "The doctor was very helpful and kind!",
        "I had to wait too long for my consultation."
    ]
    for r in analyze_batch(test_comments):
        print(r)
