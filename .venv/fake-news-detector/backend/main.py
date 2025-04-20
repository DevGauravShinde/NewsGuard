
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM

from backend.crawler import fetch_related_articles
from backend.verifier import get_verdict
from backend.summarizer import summarize_articles
from backend.utils.extract import extract_article_content

# Load models
summarizer_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

classifier_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
classifier_model = AutoModelForSequenceClassification.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class NewsInput(BaseModel):
    headline: str

# Response schema
class NewsAnalysisResponse(BaseModel):
    verdict: str
    credibility_score: float
    summary: str
    related_articles: list
    explanation: list  # List of (sentence, score)


@app.post("/analyze", response_model=NewsAnalysisResponse)
async def analyze_news(news: NewsInput):
    try:
        # Step 1: Fetch related articles (mock for now)
        related_articles = fetch_related_articles(news.headline)

        # Step 2: Fill article content using Newspaper3k
        for article in related_articles:
            article["content"] = extract_article_content(article["url"])

        # Step 3: Get verdict, score, and explanation
        verdict, credibility_score, explanation = get_verdict(news.headline, related_articles)

        # Step 4: Summarize 
        summary = summarize_articles(related_articles)

        return NewsAnalysisResponse(
         verdict=verdict,
        credibility_score=credibility_score,
        summary=summary,
        explanation=explanation,
        related_articles=related_articles
        
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))