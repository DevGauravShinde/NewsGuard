#app.py
import streamlit as st
import requests
import json

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
                # Make the API request
                response = requests.post(API_URL, json={"headline": headline})
                
                # Check if the request was successful
                if response.status_code == 200:
                    try:
                        result = response.json()
                        
                        # Check if all expected fields are present
                        required_fields = ['verdict', 'credibility_score', 'summary', 'related_articles', 'explanation']
                        missing_fields = [field for field in required_fields if field not in result]
                        
                        if missing_fields:
                            st.error(f"API response is missing required fields: {', '.join(missing_fields)}")
                            st.code(json.dumps(result, indent=2))
                        else:
                            # Display the results
                            
                            # Verdict and Score with color coding
                            st.subheader("Verdict & Score")
                            verdict = result['verdict']
                            score = result['credibility_score']
                            
                            # Color-code the verdict
                            if verdict == "Real":
                                verdict_color = "green"
                            elif verdict == "Possibly Real":
                                verdict_color = "orange"
                            else:  # Fake, Likely Fake, Unknown
                                verdict_color = "red"
                                
                            st.markdown(f"**Verdict:** <span style='color:{verdict_color}'>{verdict}</span>", unsafe_allow_html=True)
                            st.markdown(f"**Credibility Score:** `{score}`")
                            
                            # Feedback reasons if available
                            if 'feedback_reasons' in result and result['feedback_reasons']:
                                st.subheader("Why this verdict?")
                                for reason in result['feedback_reasons']:
                                    st.markdown(f"- {reason}")
                            
                            # Explanation with scores
                            if result.get("explanation"):
                                st.subheader("Evidence Analysis")
                                for sent, score in result["explanation"]:
                                    st.markdown(f"- *{sent}* (relevance: {score:.2f})")
                            
                            # Summary
                            st.subheader("📰 AI Summary")
                            st.info(result["summary"])
                            
                            # Related Articles in a more structured format
                            st.subheader("🔗 Related Articles")
                            
                            for i, article in enumerate(result["related_articles"]):
                                title = article.get("title", "No title")
                                url = article.get("url", "#")
                                source_type = article.get("source_type", "organic")
                                domain = article.get("domain", "unknown source")
                                
                                # Format source type with emoji
                                source_icon = "🔍" if source_type == "organic" else "✅" if source_type == "fact_check" else "📄"
                                
                                st.markdown(f"{i+1}. {source_icon} [{title}]({url}) - *{domain}*")
                    
                    except ValueError as e:
                        st.error(f"Error parsing API response: {e}")
                        st.code(response.text)  # Show the raw response for debugging
                else:
                    st.error(f"API request failed with status code {response.status_code}")
                    st.code(response.text)  # Show the error response
                    
            except requests.RequestException as e:
                st.error(f"Failed to connect to API: {e}")
                st.info("Make sure your FastAPI backend is running at " + API_URL)
            except Exception as e:
                st.error(f"Something went wrong: {e}")