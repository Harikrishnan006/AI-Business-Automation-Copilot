# AI Business Automation Copilot

> Upload business documents. Ask questions. Get answers with source citations.

🔗 **Live Demo:** [Click to try the app](https://huggingface.co/spaces/Harikrishnan006/ai_business_copilot)  
📁 **Type:** Portfolio Project — AI Engineering  
🛠 **Stack:** Python · Streamlit · ChromaDB · Sentence Transformers · Google Gemini · Hugging Face

---

![App Screenshot](screenshot.png)

---

## What It Does

Employees waste time searching through SOPs, contracts, and policy documents manually. This app solves that — upload any business PDF, ask a question in plain English, and get an accurate answer pulled directly from the document with the source file and page number cited.

Built on **Retrieval-Augmented Generation (RAG)** — the AI only answers from your uploaded documents, not from general knowledge.

---

## Key Features

| Feature | Description |
|---|---|
| Multi-PDF Upload | Index multiple documents at once |
| Semantic Search | Finds relevant content by meaning, not just keywords |
| Grounded Answers | AI answers only from uploaded documents |
| Source Citations | Every answer shows filename + page number |
| Confidence Score | Retrieval quality indicator per answer |
| Chat History | Full session history, scrollable |
| Document Management | View and remove indexed files |

---

## Tech Stack

| | |
|---|---|
| **Language** | Python 3.10+ |
| **UI** | Streamlit |
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) |
| **Vector DB** | ChromaDB |
| **LLM** | Google Gemini 2.5 Flash |
| **Deployment** | Hugging Face Spaces |

---

## Run Locally

```bash
git clone https://github.com/Harikrishnan006/AI-Business-Automation-Copilot.git
cd AI-Business-Automation-Copilot
pip install -r requirements.txt
# Add GEMINI_API_KEY to .env
streamlit run app.py
```

---

## Author

**Harikrishnan Venkatesan**  
AI Automation Associate · Vendasta  
Former ML Data Associate · Amazon (Rufus, Alexa+)  

[LinkedIn](https://www.linkedin.com/in/harikrishnan-venkatesan-8946a3215) · [GitHub](https://github.com/Harikrishnan006) · [Hugging Face](https://huggingface.co/Harikrishnan006)
