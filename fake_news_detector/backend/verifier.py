#verifier.py
from sentence_transformers import SentenceTransformer, util
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import re
from nltk.corpus import stopwords

# Load NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load SBERT model once
model = SentenceTransformer('all-MiniLM-L6-v2')

# Expanded list of credible news domains
CREDIBLE_DOMAINS = [
    # International sources
    'reuters.com', 'apnews.com', 'bbc.com', 'bbc.co.uk', 'npr.org', 
    'washingtonpost.com', 'nytimes.com', 'theguardian.com', 'wsj.com',
    'economist.com', 'bloomberg.com', 'cnn.com', 'nbcnews.com', 'cbsnews.com',
    'factcheck.org', 'snopes.com', 'politifact.com', 'usatoday.com',
    
    # Indian sources
    'thehindu.com', 'hindustantimes.com', 'indianexpress.com', 'ndtv.com',
    'timesnow.tv', 'timesofindia.com', 'news18.com', 'thequint.com',
    'thefederal.com', 'uniindia.com', 'punemirror.com', 'business-standard.com',
    'tribuneindia.com', 'deccanherald.com', 'newindianexpress.com', 'livemint.com',
    'thewire.in', 'theprint.in', 'scroll.in', 'firstpost.com'
]

# Keywords that often appear in sensational fake news
SENSATIONAL_KEYWORDS = [
    # Exaggeration markers
    'shocking', 'unbelievable', 'incredible', 'mind-blowing', 'bombshell', 
    'explosive', 'jaw-dropping', 'earth-shattering', 'unprecedented', 'extraordinary',
    
    # Authority undermining
    'they don\'t want you to know', 'government cover-up', 'media won\'t tell you',
    'what they\'re hiding', 'banned information', 'suppressed facts', 'censored truth',
    
    # Emotional manipulation
    'outrageous', 'scandalous', 'heart-breaking', 'terrifying', 'horrifying',
    
    # False urgency
    'urgent', 'emergency', 'alert', 'warning', 'critical', 'breaking',
    
    # Conspiracies and pseudoscience
    'conspiracy', 'illuminati', 'new world order', 'deep state', 'secret society',
    'miracle cure', 'miracle treatment', 'miracle remedy', 'ancient secret',
    
    # Clickbait phrases
    'number 7 will shock you', 'you won\'t believe', 'what happens next',
    'doctors hate this', 'one weird trick'
]

# Debunking indicators - words/phrases suggesting content is debunking a claim
DEBUNKING_INDICATORS = [
    'fake', 'false', 'not true', 'untrue', 'debunked', 'disinformation',
    'misinformation', 'hoax', 'no evidence', 'fact check', 'fact-check',
    'no proof', 'baseless', 'unfounded', 'misleading', 'fabricated',
    'manipulated', 'doctored', 'altered', 'deepfake', 'synthetic', 'ai-generated',
    'artificially created', 'not real', 'actually', 'in reality', 'in fact',
    'contrary to', 'officials deny', 'officials say no', 'never happened',
    'did not occur', 'no incident', 'no explosion', 'no attack',
    'there was no', 'does not show', 'isn\'t real'
]

# Extract domain from URL
def extract_domain(url):
    match = re.search(r'https?://(?:www\.)?([^/]+)', url)
    return match.group(1) if match else ""

# Check for sensationalist language
def contains_sensational_language(text):
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in SENSATIONAL_KEYWORDS)

# Analyze source credibility
def get_source_credibility(articles):
    if not articles:
        return 0.5  # Default to neutral when no sources available
    
    # Count credible sources
    credible_count = 0
    for article in articles:
        domain = extract_domain(article.get("url", ""))
        if any(credible in domain for credible in CREDIBLE_DOMAINS):
            credible_count += 1
    
    # If we have at least one credible source, that's good
    if credible_count > 0:
        return max(0.6, credible_count / len(articles))
    
    # If sources include major platforms that host news (but aren't in our list)
    platform_domains = ['youtube.com', 'twitter.com', 'facebook.com', 'medium.com']
    for article in articles:
        domain = extract_domain(article.get("url", ""))
        if any(platform in domain for platform in platform_domains):
            return 0.5  # Neutral for platforms
    
    return 0.4  # Slightly lower credibility for completely unknown sources

# Check for claim implausibility
def is_claim_implausible(headline):
    implausible_patterns = [
        r'\baliens\b', r'\bufo\b', r'\btime travel\b', r'\bmiracle cure\b',
        r'\bdiscovered atlantis\b', r'\bflatulence powers\b'
    ]
    
    headline_lower = headline.lower()
    return any(re.search(pattern, headline_lower) for pattern in implausible_patterns)

# Detect if articles are debunking the claim rather than supporting it
def detect_debunking_context(headline, articles):
    debunking_score = 0
    debunking_evidence = []
    
    # Normalize headline for better matching
    headline_lower = headline.lower()
    headline_words = set(re.findall(r'\b\w+\b', headline_lower))
    
    # Check each article for debunking indicators
    for article in articles:
        # Get article content
        if isinstance(article.get("content"), dict):
            text = article["content"].get("text", "").lower()
        else:
            text = str(article.get("content", "")).lower()
        
        # Skip empty content
        if not text or text.startswith("error"):
            continue
        
        # Check for debunking indicators in relation to the headline topics
        # Higher score if debunking phrases appear near headline keywords
        sentences = sent_tokenize(text)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check if the sentence mentions the headline topic
            contains_headline_topic = any(word in sentence_lower for word in headline_words if len(word) > 3)
            
            # Check if the sentence contains debunking indicators
            for indicator in DEBUNKING_INDICATORS:
                if indicator in sentence_lower:
                    # If the debunking indicator is near headline topics, this is strong evidence
                    if contains_headline_topic:
                        debunking_score += 2
                        debunking_evidence.append(sentence)
                        break
                    else:
                        debunking_score += 1
    
    # Normalize debunking score based on number of articles
    normalized_score = min(1.0, debunking_score / (len(articles) * 1.5)) if articles else 0
    
    # Check if at least one source is fact-checking specific
    fact_check_sources = 0
    for article in articles:
        title = article.get("title", "").lower()
        url = article.get("url", "").lower()
        if "fact" in title or "fact" in url or "check" in title or any(fc in url for fc in ["factcheck.", "snopes.", "politifact."]):
            fact_check_sources += 1
    
    if fact_check_sources > 0:
        normalized_score = max(normalized_score, 0.5)  # Minimum score of 0.5 if fact-checkers involved
    
    return normalized_score, list(set(debunking_evidence[:3]))  # Return top 3 unique pieces of evidence

# Analyze context of high-similarity sentences
def analyze_sentence_context(sentence, headline):
    # Convert to lowercase for better matching
    sentence_lower = sentence.lower()
    
    # Check if sentence contains debunking indicators
    contains_debunking = any(indicator in sentence_lower for indicator in DEBUNKING_INDICATORS)
    
    # Check for negations near headline keywords
    headline_words = set(re.findall(r'\b\w+\b', headline.lower()))
    significant_words = [word for word in headline_words if len(word) > 3]
    
    negation_words = ['no', 'not', 'never', 'fake', 'false', 'isn\'t', 'aren\'t', 'didn\'t', 'doesn\'t', 'don\'t']
    
    # Look for negation patterns
    for word in significant_words:
        for negation in negation_words:
            # Check if negation appears within 5 words of a headline keyword
            pattern = f'\\b{negation}\\b[^.!?]{{0,40}}\\b{word}\\b|\\b{word}\\b[^.!?]{{0,40}}\\b{negation}\\b'
            if re.search(pattern, sentence_lower):
                return -1.0  # Strong indicator of refutation/debunking
    
    if contains_debunking:
        return -0.5  # Moderate indicator of debunking
    
    return 0  # No particular contextual indicators

# Detect contradictions between sources
def detect_contradictions(articles):
    if len(articles) <= 1:
        return 0.0  # No contradictions possible with 0-1 articles
    
    # Extract main claims from each article
    article_claims = []
    for article in articles:
        # Get article content
        if isinstance(article.get("content"), dict):
            text = article["content"].get("text", "")
        else:
            text = str(article.get("content", ""))
        
        # Skip empty content
        if not text or text.startswith("Error"):
            continue
        
        # Extract first few sentences as main claims
        try:
            sentences = sent_tokenize(text)[:5]  # First 5 sentences
            if sentences:
                article_claims.append(" ".join(sentences))
        except:
            continue
    
    # Not enough claims to analyze
    if len(article_claims) <= 1:
        return 0.0
    
    # Compute pairwise similarity between claims
    contradiction_scores = []
    for i in range(len(article_claims)):
        for j in range(i+1, len(article_claims)):
            # Encode claims
            claim1_embedding = model.encode(article_claims[i], convert_to_tensor=True)
            claim2_embedding = model.encode(article_claims[j], convert_to_tensor=True)
            
            # Compute similarity
            similarity = float(util.cos_sim(claim1_embedding, claim2_embedding)[0][0].cpu().numpy())
            
            # Lower similarity might indicate contradictions
            contradiction_scores.append(1.0 - similarity)
    
    # Average contradiction score
    avg_contradiction = np.mean(contradiction_scores) if contradiction_scores else 0.0
    return min(1.0, avg_contradiction * 1.5)  # Scale up slightly but cap at 1.0

def get_verdict(headline: str, articles: list, threshold: float = 0.5):
    if not articles:
        return "Unknown", 0.5, []
    
    # Check for implausible claims first
    if is_claim_implausible(headline):
        return "Fake", 0.2, [("This headline contains claims that are scientifically implausible or extraordinary", 0.0)]

    # NEW: Check if articles are debunking the claim
    debunking_score, debunking_evidence = detect_debunking_context(headline, articles)
    
    # If strong evidence of debunking, this is likely fake news
    if debunking_score > 0.6:
        return "Fake", 0.2, [(evidence, 0.9) for evidence in debunking_evidence[:3]] + \
               [("Multiple sources indicate this news is false", 0.9)]
    
    # NEW: Check for contradictions between sources
    contradiction_score = detect_contradictions(articles)

    # Extract and process article texts
    article_texts = []
    for article in articles:
        if isinstance(article.get("content"), dict):
            text = article["content"].get("text", "")
        else:
            text = str(article.get("content", ""))
        
        # Skip error messages from web scraping
        if text and not text.startswith("Error extracting content"):
            article_texts.append(text)
    
    # Fall back to titles if no content was successfully extracted
    if not article_texts and articles:
        for article in articles:
            if article.get("title"):
                article_texts.append(article.get("title", ""))
    
    if not article_texts:
        return "Unknown", 0.5, [("Unable to analyze content from related articles", 0.0)]

    # Tokenize headline and article sentences
    sentences = []
    for text in article_texts:
        try:
            sentences.extend(sent_tokenize(text))
        except Exception:
            # Fallback for text that doesn't tokenize well
            if text:
                sentences.append(text)

    # Remove duplicates and overly short ones
    sentences = list(set(s for s in sentences if len(s.split()) > 3))

    # Compute embeddings
    headline_embedding = model.encode(headline, convert_to_tensor=True)
    
    # Skip empty sentences list
    if not sentences:
        return "Unknown", 0.5, []
        
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

    # Compute cosine similarities
    cosine_scores = util.cos_sim(headline_embedding, sentence_embeddings)[0].cpu().numpy()

    # Sort and pick top 3 most similar sentences
    top_k = min(3, len(cosine_scores))
    top_indices = np.argsort(cosine_scores)[-top_k:][::-1]
    top_sentences = [sentences[i] for i in top_indices]
    top_scores = [float(cosine_scores[i]) for i in top_indices]

    # NEW: Analyze the context of high-similarity sentences
    context_adjustments = []
    for sentence in top_sentences:
        context_score = analyze_sentence_context(sentence, headline)
        context_adjustments.append(context_score)

    # Compute content similarity score - focus on highest match but adjust for context
    max_similarity = float(np.max(cosine_scores)) if len(cosine_scores) > 0 else 0.0
    avg_similarity = float(np.mean([s for s in top_scores])) if top_scores else 0.0
    
    # Apply context adjustments to similarities
    adjusted_max = max(0.0, min(1.0, max_similarity + (context_adjustments[0] if context_adjustments else 0)))
    
    # Weight max similarity more than average
    content_similarity = 0.7 * adjusted_max + 0.3 * avg_similarity
    
    # Calculate source credibility score
    source_credibility = get_source_credibility(articles)
    
    # Check for sensational language in headline
    sensational_penalty = 0.15 if contains_sensational_language(headline) else 0.0
    
    # Calculate final score with various factors
    final_score = (
        0.4 * content_similarity +     # Content similarity weight
        0.3 * source_credibility +     # Source credibility weight
        0.2 * (1 - contradiction_score) +  # Lower score for contradictory sources
        0.1 * (1 - debunking_score) -  # Lower score if debunking indicators
        sensational_penalty            # Penalty for sensational language
    )
    
    final_score = max(0.0, min(1.0, final_score))  # Clamp between 0 and 1
    
    # Determine verdict with adjusted thresholds
    if final_score > 0.60:
        verdict = "Real"
    
    elif final_score>0.40:
        verdict="possibly real"
    else:
        verdict = "Fake"
    
    # Prepare explanation with context
    explanation = []
    for i, (sentence, score) in enumerate(zip(top_sentences, top_scores)):
        # Adjust explanation if the sentence is debunking the claim
        if context_adjustments[i] < 0:
            explanation.append((f"CLAIM: {sentence}", score))
        else:
            explanation.append((sentence, score))
    
    # Add debunking evidence if available
    for evidence in debunking_evidence[:2]:  # Top 2 debunking pieces
        if evidence not in [e[0] for e in explanation]:
            explanation.append((evidence, 0.8))
    
    # Add source credibility information
    if source_credibility > 0.6:
        explanation.append((f"Multiple credible sources are reporting this news", source_credibility))
    elif source_credibility > 0.5:
        explanation.append((f"Some credible sources are reporting this news", source_credibility))
    elif source_credibility > 0.4:
        explanation.append((f"Few established sources are covering this news", source_credibility))
    else:
        explanation.append(("No established credible sources are reporting this news", 0.0))
    
    # Add contradiction information if significant
    if contradiction_score > 0.5:
        explanation.append(("Sources present contradictory information about this claim", 0.3))
    
    # Add sensational language note if applicable
    if contains_sensational_language(headline):
        explanation.append(("The headline contains sensationalist language often associated with fake news", 0.0))

    return verdict, round(final_score, 2), explanation