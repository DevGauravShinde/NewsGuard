#summerization.py
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
import torch
import re

# Load model and tokenizer once
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Maximum length BART can handle
MAX_INPUT_LENGTH = 1024  

def clean_text(text):
    """Clean text to remove problematic patterns"""
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def summarize_articles(articles: list, max_summary_length: int = 180):
    """Create a summary from multiple articles with proper error handling and memory management"""
    summarized_chunks = []
    error_messages = []

    for article in articles:
        # Handle the case when content is a dictionary
        if isinstance(article.get("content"), dict):
            content = article["content"].get("text", "").strip()
        else:
            content = str(article.get("content", "")).strip()
            
        # Skip error messages and empty content
        if not content or content.startswith("Error extracting content") or content == "Content extraction unsuccessful":
            if content.startswith("Error extracting"):
                error_messages.append(content)
            continue

        # Clean the text
        content = clean_text(content)
        
        # Safety check - limit content size
        if len(content) > 10000:  # Trim extremely long content
            content = content[:10000]

        try:
            # Use the tokenizer's truncation to handle long inputs safely
            inputs = tokenizer(content, return_tensors="pt", truncation=True, 
                              max_length=MAX_INPUT_LENGTH-2)  # -2 for special tokens
            
            # Move inputs to the correct device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate summary
            with torch.no_grad():  # Disable gradient calculation for inference
                summary_ids = model.generate(
                    inputs["input_ids"],
                    max_length=max_summary_length,
                    min_length=min(60, len(content.split()) // 4),  # Adjust min length based on content
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )
                
            # Decode the summary
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            # Only add non-empty summaries
            if summary and len(summary) > 20:
                summarized_chunks.append(summary)
            
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            # Handle CUDA out of memory errors specially
            if "CUDA out of memory" in str(e):
                print(f"CUDA out of memory error, trying with smaller input")
                try:
                    # Try again with a much smaller input
                    shorter_content = content[:3000]  # First 3000 chars only
                    inputs = tokenizer(shorter_content, return_tensors="pt", truncation=True, 
                                      max_length=512)  # Much shorter max length
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        summary_ids = model.generate(
                            inputs["input_ids"],
                            max_length=max_summary_length // 2,
                            min_length=30,
                            num_beams=2  # Reduce beam search complexity
                        )
                    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                    if summary:
                        summarized_chunks.append(summary)
                        
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as inner_e:
                    print(f"[Inner summarization error] {inner_e}")
            else:
                print(f"[Summarization error] {e}")
                
        except Exception as e:
            print(f"[Summarization error] {e}")
            
    # Combine summaries with error messages
    final_summary = "\n\n".join(summarized_chunks)
    
    # Add error messages if they exist but still have some summary
    if error_messages and final_summary:
        final_summary = "Note: Some content couldn't be accessed.\n\n" + final_summary
    
    # If no summary was generated but we have error messages
    elif error_messages and not final_summary:
        return "Unable to summarize content. Some sources were inaccessible."
        
    return final_summary.strip() if final_summary else "No content available to summarize."