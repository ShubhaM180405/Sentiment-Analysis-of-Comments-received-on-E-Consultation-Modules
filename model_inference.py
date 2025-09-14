# model_inference.py
from transformers import pipeline

# Load sentiment analysis pipeline with 3 labels: NEGATIVE, NEUTRAL, POSITIVE
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

def analyze_sentiment(comment: str):
    """
    Analyze sentiment of a single comment.
    Returns label and confidence score.
    """
    result = sentiment_pipeline(comment[:512])[0]  # truncate if longer than 512 tokens
    return {"comment": comment, "label": result["label"], "score": round(result["score"], 4)}

def analyze_batch(comments: list):
    """
    Analyze sentiment of a batch of comments.
    Returns list of dicts with label + score.
    """
    results = sentiment_pipeline(comments, truncation=True)
    output = []
    for comment, res in zip(comments, results):
        output.append({"comment": comment, "label": res["label"], "score": round(res["score"], 4)})
    return output
