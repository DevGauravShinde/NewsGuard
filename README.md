![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)

# 🛡️ NewsGuard - Fake News Analyzer

NewsGuard is an AI-powered tool that analyzes news headlines to detect potential misinformation. It uses **BART for summarization** and **SBERT for semantic similarity analysis** to cross-reference the headline with real-time news articles from credible sources.

## 🔍 Features :

- ✅ Detects whether a headline is **Real**, **Possibly Real**, or **Fake**
- 🧠 Uses **semantic similarity (SBERT)** to compare claims with real-world sources
- 📉 Penalizes headlines using **sensationalist language**
- 🔗 Fetches related news articles using **SerpAPI + Newspaper3k**
- 📝 Provides AI-generated summaries and evidence-based explanations
- 🌐 Supports fact-check detection from credible fact-checking domains

---

# 🧪 Technologies Used :

- **FastAPI** – Backend API
- **Streamlit** – Frontend web interface
- **SBERT (all-MiniLM-L6-v2)** – Semantic similarity
- **BART** – Summarization
- **Newspaper3k** – Article scraping
- **SerpAPI** – Real-time Google Search results
- **NLTK** – Text preprocessing

---

# 🚀 Getting Started

# 1. Clone the repo :

``` bash
git clone https://github.com/DevGauravShinde/NewsGuard.git
cd NewsGuard/fake_news_detector
```

# 2.Setup environment :
python -m venv .venv
 For Windows:
.venv\Scripts\activate
 For Mac/Linux:
source .venv/bin/activate


# 3.Set Your API key :
Create a .env file in the fake_news_detector folder:
SERPAPI_KEY=your_serpapi_key_here

# 4. Run the backend :
uvicorn backend.main:app --reload

# 5. Run the frontend :
streamlit run streamlit_app/app.py

# 📁 Project Structure

NewsGuard
├─ fake_news_detector
│  ├─ backend
│  │  ├─ crawler.py
│  │  ├─ main.py
│  │  ├─ models
│  │  │  └─ sbert_model.pkl
│  │  ├─ summarizer.py
│  │  ├─ utils
│  │  │  └─ extract.py
│  │  ├─ verifier.py
│  │  └─ __init__.py
│  ├─ extension
│  ├─ frontend
│  ├─ requirement.txt
│  ├─ streamlit_app
│  │  └─ app.py
│  └─ __init__.py
├─ .env
└─ README.md


# 📜 License
This project is licensed under the MIT License © Gaurav Shinde.

🤝 Contributions
PRs and suggestions are welcome. Let’s build a smarter internet together. 🧠

---

## 👤 About the Author

**Gaurav Valmik Shinde**  
🎓 BTech CSE (AI & ML) Student at Vishwakarma Institute of Information Technology, Pune  
💡 Passionate about Generative AI (GenAI), LangChain, LLMs, and building intelligent systems that solve real-world problems  
💼 Experienced in full-stack AI development using NLP, FastAPI, Streamlit, and Python  
📫 Reach me at: [gvshinde2004@gmail.com](mailto:gvshinde2004@gmail.com)  
🌐 GitHub: [DevGauravShinde](https://github.com/DevGauravShinde) | [LinkedIn](https://www.linkedin.com/in/gaurav-shinde-cs/)
