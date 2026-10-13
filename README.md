# DocGenius – AI-Powered PDF Assistant

A Streamlit app that lets you upload PDFs and have a conversation with them. Ask questions in plain English and get answers pulled directly from the document, with citations showing which page the information came from.

Powered by Google Gemini 1.5 Flash and LangChain.

---

## Background

This started as a fork of [KalyanMurapaka45's DocGenius project](https://github.com/KalyanMurapaka45/DocGenius-Revolutionizing-PDFs-with-AI), which was built on OpenAI and an older version of LangChain. I rebuilt most of it — swapped the entire AI stack over to Google Gemini (free API, no credit card), updated the LangChain implementation to use the current LCEL approach since the old chain/memory APIs were removed in LangChain 1.x, and added a bunch of features that weren't there originally.

---

## What it does

- Upload one or more PDFs and query across all of them at once
- Remembers the conversation so follow-up questions work naturally
- Generates a summary of the document automatically after upload
- Shows which file and page each answer came from
- Export the full chat session as a .txt or .json file
- Temperature slider to control how creative or literal the responses are
- API key can be entered directly in the sidebar — no .env file needed to run it

---

## How it works

When you upload a PDF, the app extracts the text and splits it into overlapping chunks. Those chunks get embedded using Google's embedding model and stored in a local FAISS index. When you ask a question, it searches the index for the most relevant chunks, passes them as context to Gemini along with the recent conversation history, and returns an answer.

The source citations work by tracking which character range in the full text each chunk came from, then mapping that back to the original file and page number.

---

## Tech

- [Google Gemini 1.5 Flash](https://deepmind.google/technologies/gemini/) — handles Q&A and document summarization
- [LangChain](https://www.langchain.com/) — LCEL chain for prompt/LLM/parser composition
- [langchain-google-genai](https://python.langchain.com/docs/integrations/chat/google_generative_ai/) — Gemini + embeddings integration
- [FAISS](https://faiss.ai/) — local vector similarity search, no external database needed
- [PyPDF2](https://pypdf2.readthedocs.io/) — PDF text extraction
- [Streamlit](https://streamlit.io/) — UI framework

---

## Setup

You'll need a Google API key. Get one for free at https://aistudio.google.com/app/apikey — no credit card required.

```bash
git clone https://github.com/deancello11/AI-Powered-PDF-Assistant.git
cd AI-Powered-PDF-Assistant

python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS / Linux

pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_key_here
```

Or just paste the key into the sidebar when the app is running.

```bash
streamlit run app.py
```

---

## Project structure

```
├── app.py                 # main application
├── requirements.txt
├── .env.example           # template for the API key
└── DocGenius/
    └── PDFChat.py         # stripped-down single-PDF version, kept as reference
```

---

## License

GPL v3. See LICENSE for details.

---

## Credit

Original project by [KalyanMurapaka45](https://github.com/KalyanMurapaka45).
