import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from model_inference import predict_sentiment

st.set_page_config(page_title="E-Consultation Sentiment Analysis", layout="wide")

st.title("💬 Sentiment Analysis of E-Consultation Comments")
st.write("Analyze feedback using a BERT model in real-time!")

# Single comment analysis
st.header("🔹 Single Comment Analysis")
user_input = st.text_area("Enter a comment here:")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        label, score = predict_sentiment(user_input)
        st.success(f"**Sentiment:** {label} | **Confidence:** {score:.2f}")
    else:
        st.warning("⚠️ Please enter a comment.")

# Bulk analysis
st.header("📂 Bulk Comment Analysis (CSV Upload)")
uploaded_file = st.file_uploader("Upload a CSV file with a 'comment' column", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "comment" not in df.columns:
        st.error("CSV must contain a 'comment' column.")
    else:
        # Run sentiment analysis
        results = [predict_sentiment(text) for text in df["comment"].astype(str)]
        df["Sentiment"], df["Confidence"] = zip(*results)

        st.subheader("📋 Results")
        st.dataframe(df, use_container_width=True, height=600)  # ✅ Scrollable table

        # Plot distribution
        st.subheader("📊 Sentiment Distribution")
        sentiment_counts = df["Sentiment"].value_counts()

        fig, ax = plt.subplots()
        sentiment_counts.plot(kind="bar", ax=ax)
        ax.set_title("Sentiment Distribution")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        # WordCloud
        st.subheader("☁️ WordCloud of Comments")
        text = " ".join(df["comment"].astype(str))
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
