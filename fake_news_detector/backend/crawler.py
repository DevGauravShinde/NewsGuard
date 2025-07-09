#crawler.py
import requests
import os
from newspaper import Article, build
import time
import re
from urllib.parse import urlparse
import random
from dotenv import load_dotenv
load_dotenv()

SERPAPI_KEY = os.getenv("SERPAPI_KEY")  # Set this in your environment

# List of common user agents to rotate through
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
]

def extract_domain(url):
    """Extract the domain from a URL"""
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    if domain.startswith('www.'):
        domain = domain[4:]
    return domain

def extract_article_content(url: str, max_retries=2):
    """Extract article content and metadata using Newspaper3k with retry logic and user agent rotation"""
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            # Configure article with a random user agent
            article = Article(url)
            article.config.browser_user_agent = random.choice(USER_AGENTS)
            article.config.request_timeout = 10  # Increase timeout 
            
            article.download()
            article.parse()
            
            # Skip NLP if text extraction was successful but short
            if len(article.text) > 100:
                try:
                    article.nlp()  # This performs NLP to extract keywords, summary
                except Exception as nlp_err:
                    # Continue even if NLP fails
                    print(f"NLP failed but continuing: {str(nlp_err)}")
            
            # Get additional metadata
            source_domain = extract_domain(url)
            
            # Create a richer content object
            content_data = {
                "text": article.text if article.text else "No content extracted",
                "summary": getattr(article, 'summary', ''),
                "keywords": getattr(article, 'keywords', []),
                "publish_date": str(article.publish_date) if article.publish_date else None,
                "source_domain": source_domain,
                "authors": getattr(article, 'authors', [])
            }
            
            return content_data
        
        except Exception as e:
            retry_count += 1
            if retry_count <= max_retries:
                # Wait before retry (exponential backoff)
                time.sleep(2 ** retry_count)  
                print(f"Retrying {url} ({retry_count}/{max_retries})")
            else:
                # If all retries fail, return a simple error message rather than the full exception
                return {"text": f"Content extraction unsuccessful"}

def fetch_related_articles(headline: str, num_results: int = 8):
    """Fetch related articles with improved source diversity and error handling"""
    if not SERPAPI_KEY:
        # Mock fallback if no key is available
        return [
            {
                "title": "Fact-check: Related to " + headline,
                "url": "https://www.reliable-source.com/article1",
                "content": {"text": "This is sample mock content for development."}
            },
            {
                "title": "Why claims about " + headline.split()[0] + " are circulating online",
                "url": "https://www.fact-check-site.org/article2",
                "content": {"text": "This is another sample mock content for development."}
            }
        ]

    # Normal search
    params = {
        "q": headline,
        "location": "United States",
        "hl": "en",
        "gl": "us",
        "num": num_results,
        "api_key": SERPAPI_KEY,
        "engine": "google"
    }

    normal_articles = []
    
    try:
        response = requests.get("https://serpapi.com/search", params=params)
        if response.status_code == 200:
            data = response.json()
            for result in data.get("organic_results", []):
                title = result.get("title")
                url = result.get("link")
                
                if title and url:
                    normal_articles.append({
                        "title": title,
                        "url": url,
                        "source_type": "organic"
                    })
    except Exception as e:
        print(f"Error fetching normal search results: {str(e)}")
    
    # Fact-check specific search
    fact_check_params = params.copy()
    fact_check_params["q"] = f"fact check {headline}"
    
    fact_check_articles = []
    
    try:
        response = requests.get("https://serpapi.com/search", params=fact_check_params)
        if response.status_code == 200:
            data = response.json()
            for result in data.get("organic_results", []):
                title = result.get("title")
                url = result.get("link")
                
                if title and url and ("fact" in title.lower() or "check" in title.lower()):
                    fact_check_articles.append({
                        "title": title,
                        "url": url,
                        "source_type": "fact_check"
                    })
    except Exception as e:
        print(f"Error fetching fact-check search results: {str(e)}")
    
    # Combine results, prioritizing fact-checks but maintaining diversity
    combined_articles = []
    
    # Always include fact checks first (they're more valuable)
    for article in fact_check_articles[:3]:  # Include up to 3 fact checks
        content = extract_article_content(article["url"])
        article["content"] = content
        combined_articles.append(article)
    
    # Then add normal articles until we reach our desired count
    domains_seen = set(extract_domain(article["url"]) for article in combined_articles)
    
    for article in normal_articles:
        domain = extract_domain(article["url"])
        # Add some domain diversity
        if domain not in domains_seen or len(combined_articles) < 5:
            content = extract_article_content(article["url"])
            article["content"] = content
            combined_articles.append(article)
            domains_seen.add(domain)
            
        if len(combined_articles) >= num_results:
            break
    
    return combined_articles