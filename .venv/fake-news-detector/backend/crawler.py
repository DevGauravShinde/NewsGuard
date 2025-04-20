
import requests
import os
from newspaper import Article

SERPAPI_KEY = os.getenv("SERPAPI_KEY")  # Set this in your environment

def extract_article_content(url: str):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return f"Error extracting content: {str(e)}"

def fetch_related_articles(headline: str, num_results: int = 5):
    if not SERPAPI_KEY:
        # Mock fallback if no key is available
        return [
            {
                "title": "Fact-check: Viral claim about X is false",
                "url": "https://www.reliable-source.com/article1",
                "content": "This is sample mock content for development."
            }
        ]

    params = {
        "q": headline,
        "location": "India",  # or user location
        "hl": "en",
        "gl": "in",
        "num": num_results,
        "api_key": SERPAPI_KEY,
        "engine": "google"
    }

    response = requests.get("https://serpapi.com/search", params=params)
    if response.status_code != 200:
        raise Exception("Failed to fetch from SerpAPI")

    data = response.json()
    articles = []

    # Parse organic search results
    for result in data.get("organic_results", [])[:num_results]:
        title = result.get("title")
        url = result.get("link")

        if title and url:
            # Extract article content using Newspaper3k or another method
            content = extract_article_content(url)
            articles.append({
                "title": title,
                "url": url,
                "content": content  # Fill with real content extracted from the URL
            })

    return articles