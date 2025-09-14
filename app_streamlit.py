# app_streamlit.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from model_inference import analyze_sentiment, analyze_batch

# Page config
st.set_page_config(page_title="E-Consultation Sentiment Analysis", layout="wide")

st.title("💬 Sentiment Analysis of E-Consultation Comments")
st.write("Analyze patient feedback in real-time.")

# Sidebar options
st.sidebar.title("Options")
mode = st.sidebar.radio("Choose Input Mode:", ["Single Comment", "Upload File"])

if mode == "Single Comment":
    comment = st.text_area("✍️ Enter a comment for analysis:")
    if st.button("Analyze"):
        if comment.strip():
            result = analyze_sentiment(comment)
            st.subheader("🔎 Result")
            st.write(f"**Sentiment:** {result['label']}  |  **Confidence:** {result['score']}")
        else:
            st.warning("Please enter a comment.")

elif mode == "Upload File":
    uploaded_file = st.file_uploader("📂 Upload a CSV file with a 'comment' column", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = [col.lower() for col in df.columns]
        comment_col = None
        if "comment" in df.columns:
            comment_col = "comment"
        elif "comments" in df.columns:
            comment_col = "comments"
        if comment_col is None:
            st.error("CSV must contain a 'comment' column.")
        else:
            st.write("✅ File uploaded successfully!")
            results = analyze_batch(df[comment_col].tolist())
            results_df = pd.DataFrame(results)
            
            st.subheader("📋 Results")
            st.dataframe(results_df, use_container_width=True, height=600)  # ✅ scrollable

            # Sentiment distribution
            st.subheader("📊 Sentiment Distribution")
            sentiment_counts = results_df['label'].value_counts()
            fig, ax = plt.subplots()
            sentiment_counts.plot(kind="bar", ax=ax, color=["green", "blue", "red"])
            plt.xticks(rotation=0)
            st.pyplot(fig)

            # Wordcloud
            st.subheader("☁️ WordCloud of Feedback")
            text = " ".join(results_df['comment'])
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
            fig_wc, ax_wc = plt.subplots()
            ax_wc.imshow(wordcloud, interpolation="bilinear")
            ax_wc.axis("off")
            st.pyplot(fig_wc)
