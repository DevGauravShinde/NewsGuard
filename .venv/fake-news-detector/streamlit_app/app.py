import streamlit as st
import requests

API_URL = "http://localhost:8000/analyze"  # change if deployed elsewhere

st.set_page_config(page_title="Fake News Analyzer", layout="wide")
st.title("📰 Fake News Analyzer")
st.markdown("Analyze a news headline to detect misinformation using BART + SBERT")

# Headline input
headline = st.text_input("🔍 Enter a news headline")

if st.button("Analyze"):
    if not headline.strip():
        st.warning("Please enter a headline.")
    else:
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(API_URL, json={"headline": headline})
                result = response.json()

                # Verdict and Score
                st.subheader("Verdict & Score")
                st.markdown(f"**Verdict:** `{result['verdict']}`")
                st.markdown(f"**Credibility Score:** `{result['credibility_score']}`")

                # Explanation
                if result.get("explanation"):
                    st.subheader(" Explanation (Supporting Sentences)")
                    for sent in result["explanation"]:
                        st.markdown(f"- *{sent}*")

                # Summary
                st.subheader("📰 AI Summary")
                st.info(result["summary"])

                # Related Articles
                st.subheader("🔗 Related Articles")
                for article in result["related_articles"]:
                    title = article.get("title", "No title")
                    url = article.get("url", "#")
                    st.markdown(f"- [{title}]({url})")

            except Exception as e:
                st.error(f"Something went wrong: {e}")
