#main.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import logging

from fake_news_detector.backend.crawler import fetch_related_articles

from fake_news_detector.backend.verifier import get_verdict
from fake_news_detector.backend.summarizer import summarize_articles

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    feedback_reasons: list = []  # Additional context about the verdict

@app.post("/analyze", response_model=NewsAnalysisResponse)
async def analyze_news(news: NewsInput):
    try:
        headline = news.headline.strip()
        if not headline:
            raise HTTPException(status_code=400, detail="Headline cannot be empty")
            
        logger.info(f"Analyzing headline: {headline}")
        
        # Step 1: Fetch related articles
        related_articles = fetch_related_articles(headline)
        
        # Process articles to ensure content is accessible for verifier
        processed_articles = []
        for article in related_articles:
            processed_article = article.copy()
            
            # If content is a dict, keep it as is
            if isinstance(article.get("content"), dict):
                processed_article["content"] = article["content"].get("text", "")
            # Otherwise, ensure content is a string
            elif article.get("content") is not None:
                processed_article["content"] = str(article["content"])
            else:
                processed_article["content"] = ""
                
            processed_articles.append(processed_article)
        
        if not processed_articles:
            logger.warning(f"No related articles found for: {headline}")
            return NewsAnalysisResponse(
                verdict="Unknown",
                credibility_score=0.0,
                summary="Unable to find related articles to verify this claim.",
                related_articles=[],
                explanation=[("No supporting or refuting evidence found", 0.0)],
                feedback_reasons=["No search results found"]
            )
        
        # Step 2: Get verdict, score, and explanation
        verdict, credibility_score, explanation = get_verdict(headline, processed_articles)
        
        # Step 3: Generate summary from the related articles
        summary = summarize_articles(related_articles)
        
        # Step 4: Prepare feedback reasons based on verdict
        feedback_reasons = []
        
        if verdict == "Fake" or verdict == "Likely Fake":
            feedback_reasons.append("Low content similarity with credible sources")
            
            # Check if there are fact checks specifically refuting the claim
            fact_checks = [art for art in related_articles if art.get("source_type") == "fact_check"]
            if fact_checks:
                feedback_reasons.append("Fact-checking sites have analyzed this claim")
                
        elif verdict == "Possibly Real":
            feedback_reasons.append("Some supporting evidence found, but not conclusive")
            
        elif verdict == "Real":
            feedback_reasons.append("Consistent reporting across multiple sources")
        
        # Format related articles for response
        formatted_articles = []
        for art in related_articles:
            formatted_art = {
                "title": art.get("title", ""),
                "url": art.get("url", ""),
                "source_type": art.get("source_type", "organic")
            }
            
            # Add source domain if available
            if isinstance(art.get("content"), dict) and "source_domain" in art["content"]:
                formatted_art["domain"] = art["content"]["source_domain"]
                
            formatted_articles.append(formatted_art)
        
        return NewsAnalysisResponse(
            verdict=verdict,
            credibility_score=credibility_score,
            summary=summary,
            explanation=explanation,
            related_articles=formatted_articles,
            feedback_reasons=feedback_reasons
        )

    except Exception as e:
        logger.error(f"Error analyzing headline: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
