from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
import torch

# Load model and tokenizer once
device = 0 if torch.cuda.is_available() else -1
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

MAX_INPUT_LENGTH = 1024  # max tokens BART can handle

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

def chunk_text(text, max_length=1024, overlap=100):
    tokens = tokenizer.encode(text)
    chunks = []

    for i in range(0, len(tokens), max_length - overlap):
        chunk = tokens[i:i + max_length]
        decoded_chunk = tokenizer.decode(chunk, skip_special_tokens=True)
        chunks.append(decoded_chunk)

    return chunks

def summarize_articles(articles: list, max_summary_length: int = 180):
    summarized_chunks = []

    for article in articles:
        content = article.get("content", "").strip()
        if not content:
            continue

        chunks = chunk_text(content)
        article_summary_parts = []

        for chunk in chunks:
            try:
                summary = summarizer(
                    chunk,
                    max_length=max_summary_length,
                    min_length=60,
                    do_sample=False
                )[0]['summary_text']
                article_summary_parts.append(summary)
            except Exception as e:
                print(f"[Summarization error] {e}")

        if article_summary_parts:
            summarized_chunks.append(" ".join(article_summary_parts))

    final_summary = "\n\n".join(summarized_chunks)
    return final_summary.strip() if final_summary else "No content available to summarize."
