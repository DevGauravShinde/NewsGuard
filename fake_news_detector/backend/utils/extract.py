#extract.py
from newspaper import Article

def extract_article_content(url: str) -> str:
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Error extracting content for {url}: {str(e)}")
        return ""