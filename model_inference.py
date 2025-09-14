# model_inference.py
from transformers import pipeline

# Load 3-class sentiment model
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# Map model outputs to readable labels
label_mapping = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

def analyze_sentiment(comment: str):
    """
    Analyze sentiment of a single comment.
    Returns label and confidence score.
    """
    result = sentiment_pipeline(comment[:512])[0]  # truncate long comments
    return {
        "comment": comment,
        "label": label_mapping[result["label"]],
        "score": round(result["score"], 4)
    }

def analyze_batch(comments: list):
    """
    Analyze sentiment of a batch of comments.
    Returns list of dicts with label + score.
    """
    results = sentiment_pipeline(comments, truncation=True)
    output = []
    for comment, res in zip(comments, results):
        output.append({
            "comment": comment,
            "label": label_mapping[res["label"]],
            "score": round(res["score"], 4)
        })
    return output
