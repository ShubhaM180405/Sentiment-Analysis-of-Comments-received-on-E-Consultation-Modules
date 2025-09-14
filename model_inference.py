from transformers import pipeline
import numpy as np

# Load 3-class sentiment model
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", return_all_scores=True)

label_mapping = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

def get_custom_label(scores):
    # Convert to dict {label: score}
    score_dict = {label_mapping[i]: s['score'] for i, s in enumerate(scores)}

    # Find top label
    top_label = max(score_dict, key=score_dict.get)
    top_score = score_dict[top_label]

    # If neutral is close, mark as leaning
    if score_dict["Neutral"] > 0.3 and top_label != "Neutral":
        return f"Neutral (but leaning towards {top_label} side)", round(score_dict["Neutral"], 4)

    return top_label, round(top_score, 4)


def analyze_sentiment(comment: str):
    scores = sentiment_pipeline(comment[:512])[0]
    label, confidence = get_custom_label(scores)
    return {"comment": comment, "label": label, "score": confidence}


def analyze_batch(comments: list):
    results = sentiment_pipeline(comments, truncation=True)
    output = []
    for comment, scores in zip(comments, results):
        label, confidence = get_custom_label(scores)
        output.append({"comment": comment, "label": label, "score": confidence})
    return output
