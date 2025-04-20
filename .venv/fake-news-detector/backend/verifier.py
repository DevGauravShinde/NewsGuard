from sentence_transformers import SentenceTransformer, util
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize

# Load NLTK resources
nltk.download('punkt')

# Load SBERT model once
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_verdict(headline: str, articles: list, threshold: float = 0.5):
    if not articles:
        return "Unknown", 0.0, []

    article_texts = [article["content"] for article in articles if article["content"]]

    if not article_texts:
        return "Unknown", 0.0, []

    # Tokenize headline and article sentences
    sentences = []
    for text in article_texts:
        sentences.extend(sent_tokenize(text))

    # Remove duplicates and overly short ones
    sentences = list(set(s for s in sentences if len(s.split()) > 5))

    # Compute embeddings
    headline_embedding = model.encode(headline, convert_to_tensor=True)
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

    # Compute cosine similarities
    cosine_scores = util.cos_sim(headline_embedding, sentence_embeddings)[0].cpu().numpy()

    # Sort and pick top 3 most similar sentences
    top_k = min(3, len(cosine_scores))
    top_indices = np.argsort(cosine_scores)[-top_k:][::-1]
    top_sentences = [sentences[i] for i in top_indices]
    top_scores = [float(cosine_scores[i]) for i in top_indices]

    # Compute overall similarity score
    best_score = float(np.max(cosine_scores)) if len(cosine_scores) > 0 else 0.0

    # Determine verdict
    if best_score > threshold:
        verdict = "Real"
    elif best_score > 0.3:
        verdict = "Possibly Real"
    else:
        verdict = "Fake"

    return verdict, round(best_score, 2), list(zip(top_sentences, top_scores))
