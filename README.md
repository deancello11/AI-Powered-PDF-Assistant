# 📄 DocGenius – AI-Powered PDF Assistant

> Chat with your documents. Upload PDFs, ask questions in plain English, and get cited answers powered by **Google Gemini 1.5 Flash**.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.43-red?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-1.x-green)
![Gemini](https://img.shields.io/badge/Google%20Gemini-1.5%20Flash-orange?logo=google)
![License](https://img.shields.io/badge/License-GPL%20v3-lightgrey)

---

## About This Project

This started as a fork of an OpenAI-based PDF Q&A app. I rebuilt it from the ground up — swapping the entire AI stack to Google Gemini, modernizing the LangChain implementation to use the current LCEL pipeline (the old chain/memory APIs were removed in LangChain 1.x), and adding a full suite of new features to turn it into something portfolio-worthy.

The app lets you upload one or more PDFs, processes them into a FAISS vector store using Google's embedding model, and then lets you have a full conversation about the content — with source citations, document summaries, chat export, and a custom dark UI.

---

## What I Changed

The original project used OpenAI embeddings and GPT for Q&A, had a single-file upload, no chat memory, a hardcoded file path that broke on every machine, and default Streamlit styling.

Here's what I rebuilt:

**AI Stack**
- Replaced OpenAI entirely with **Google Gemini 1.5 Flash** — free tier available, no credit card needed
- Replaced OpenAI embeddings with **Google's `embedding-001` model** via `langchain-google-genai`
- Rewrote the chain using **LCEL** (LangChain Expression Language), the modern approach that works with LangChain 1.x:
  ```python
  chain = prompt | llm | StrOutputParser()
  ```

**PDF Processing**
- Added **multi-PDF support** — upload and query across several documents at once
- Added **MD5 fingerprinting** so re-uploading the same file doesn't reprocess it
- Switched from `CharacterTextSplitter` to `RecursiveCharacterTextSplitter` for smarter chunking
- Added **page-level metadata tracking** so every answer can be traced back to a specific file and page

**New Features**
- Conversational memory — follow-up questions work naturally
- Auto document summary generated on upload
- Source citations on every answer (file name + page number)
- Export chat history as `.txt` or `.json`
- Quick-question chips for common tasks
- Temperature slider to tune response creativity
- Document stats panel (files, pages, word count)
- API key input directly in the UI — no `.env` file required to run

**UI**
- Full custom dark theme with CSS — gradient header, animated chat bubbles, source citation cards
- Proper landing page with feature cards when no documents are loaded

**Code Quality**
- Removed hardcoded absolute file paths
- Added `.env.example` template
- Updated `.gitignore` to exclude secrets and test PDFs
- Rewrote this README

---

## Features

| Feature | Description |
|---|---|
| 📚 Multi-PDF Upload | Load and query multiple PDFs simultaneously |
| 🧠 Conversational Memory | Remembers context across follow-up questions |
| 📋 Auto Document Summary | Structured summary generated on upload |
| 📌 Source Citations | Every answer links back to the exact file and page |
| 💾 Export Chat | Download your session as `.txt` or `.json` |
| 🎛️ Tunable Responses | Adjust AI creativity with a temperature slider |
| ⚡ Quick-Question Chips | One-click prompts for common tasks |
| 🎨 Custom Dark UI | Clean interface built with custom CSS |

---

## Architecture

```
User uploads PDF(s)
        │
        ▼
   PyPDF2 extracts text + page metadata
        │
        ▼
RecursiveCharacterTextSplitter
   chunks the text (~1500 chars with overlap)
        │
        ▼
Google Generative AI Embeddings
   (models/embedding-001)
        │
        ▼
    FAISS vector store (local, in-memory)
        │
  User asks a question
        │
        ▼
 Similarity search → top 5 chunks retrieved
        │
        ▼
LCEL chain: ChatPromptTemplate | Gemini 1.5 Flash | StrOutputParser
        │
        ▼
   Answer + Source citations displayed
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| [Google Gemini 1.5 Flash](https://deepmind.google/technologies/gemini/) | LLM for Q&A and summarization |
| [LangChain](https://www.langchain.com/) | LCEL chain orchestration |
| [langchain-google-genai](https://python.langchain.com/docs/integrations/chat/google_generative_ai/) | Gemini + embeddings integration |
| [FAISS](https://faiss.ai/) | Local vector similarity search |
| [PyPDF2](https://pypdf2.readthedocs.io/) | PDF text extraction |
| [Streamlit](https://streamlit.io/) | Web UI framework |

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/DocGenius-Revolutionizing-PDFs-with-AI.git
cd DocGenius-Revolutionizing-PDFs-with-AI
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your API key

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_google_api_key_here
```

Get a free key at [Google AI Studio](https://aistudio.google.com/app/apikey) — no credit card needed.

Alternatively, paste the key directly into the sidebar input when the app is running.

### 5. Run the app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## Project Structure

```
DocGenius-Revolutionizing-PDFs-with-AI/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Pinned Python dependencies
├── .gitignore
├── README.md
└── DocGenius/
    └── PDFChat.py          # Lightweight single-PDF version (legacy reference)
```

---

## License

Distributed under the **GNU General Public License v3.0**. See `LICENSE` for details.

---

## Acknowledgements

- Original project concept by [KalyanMurapaka45](https://github.com/KalyanMurapaka45/DocGenius-Revolutionizing-PDFs-with-AI)
- [Google AI Studio](https://aistudio.google.com/) for the Gemini API
- The LangChain open-source community
