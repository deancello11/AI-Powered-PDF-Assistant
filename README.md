# DocGenius – AI-Powered PDF Assistant

A Streamlit app that lets you upload PDFs and have a conversation with them. Ask questions in plain English and get answers pulled directly from the document, with citations showing which page the information came from.

Built on Google Gemini 2.5 Flash and LangChain.

---

## Background

This started as a fork of [KalyanMurapaka45's DocGenius project](https://github.com/KalyanMurapaka45/DocGenius-Revolutionizing-PDFs-with-AI), which was built on OpenAI and an older version of LangChain. I rebuilt most of it — swapped the entire AI stack over to Google Gemini (free API, no credit card), updated the LangChain implementation to use the current LCEL approach since the old chain/memory APIs were removed in LangChain 1.x, and added features that weren't there originally.

---

## What it does

- Upload one or more PDFs and query across all of them at once
- Remembers the conversation so follow-up questions work naturally
- Generates a summary of the full document after upload, not just the first page
- Shows which file and page each answer came from (approximate — see notes below)
- Export the full chat session as a .txt or .json file
- Temperature slider to control how creative or literal the responses are
- API key can be entered directly in the sidebar, no .env file needed to run it

---

## How it works

When you upload a PDF, the app extracts the text page by page and splits it into overlapping chunks. Those chunks get embedded using Google's embedding model and stored in a local FAISS index. When you ask a question, it retrieves the 5 most relevant chunks, passes them as context to Gemini along with the recent conversation history, and returns an answer.

**Chunk size** is 1500 characters with 200 overlap. The overlap stops answers from falling through the gap between two adjacent chunks. 1500 chars is roughly a long paragraph — small enough for precise retrieval, large enough that the model gets enough context per chunk to give a useful answer.

**k=5** means five chunks are retrieved per question. That's usually enough to cover multi-part answers without flooding the context window with irrelevant text.

**Temperature** defaults to 0.3 — low enough to keep answers grounded in the document, high enough to write naturally. Setting it to 0 makes it very literal; setting it higher is useful for creative tasks but risks the model going off-script.

**Summarization** works in rolling 8,000-character sections rather than just the first chunk, so longer documents and multi-PDF uploads get a full overview rather than a summary of only the first file.

---

## Notes on citations

Page numbers are approximate. The app finds where each retrieved chunk sits in the full extracted text by character position, then maps that back to the page it came from. This works well in most cases but has two edge cases: if the same passage appears more than once, it will always match the first occurrence; and a chunk that spans a page boundary will show the page where it starts. A future improvement would be to attach page numbers to each chunk at split time.

---

## Security

All user input and model output inserted into HTML is run through `html.escape()` before rendering. This prevents a malicious PDF from getting the model to output HTML tags — like an image tag pointing at an external server — that the browser would silently render. It is a small thing but worth noting given the app ingests untrusted documents.

Conversation history is passed to the prompt via LangChain's `MessagesPlaceholder` rather than as raw strings. This prevents a bug where curly braces in answers (code snippets, JSON, etc.) get interpreted as template variables and crash the next request.

---

## Tech

- [Google Gemini 2.5 Flash](https://deepmind.google/technologies/gemini/) — handles Q&A and summarization
- [LangChain](https://www.langchain.com/) — LCEL chain for prompt/LLM/parser composition
- [langchain-google-genai](https://python.langchain.com/docs/integrations/chat/google_generative_ai/) — Gemini + embeddings integration
- [FAISS](https://faiss.ai/) — local vector similarity search, no external database needed
- [PyPDF2](https://pypdf2.readthedocs.io/) — PDF text extraction
- [Streamlit](https://streamlit.io/) — UI framework

---

## Setup

You will need a Google API key. Get one free at https://aistudio.google.com/app/apikey — no credit card required.

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

Or paste the key into the sidebar when the app is running.

```bash
streamlit run app.py
```

---

## Known limitations and what I would improve

- Adding a new file re-embeds all documents from scratch. The right fix is to cache embeddings per file and only embed new ones, then merge the indexes.
- Page number attribution is approximate. Attaching page metadata to each chunk at split time would make citations exact.
- No streaming. Responses appear all at once after the full generation completes. LangChain supports streaming with `.stream()` and Streamlit can render it with `st.write_stream`, which would make long answers feel faster.

---

## Project structure

```
├── app.py                 # main application
├── requirements.txt
├── .env.example           # template showing which environment variables are needed
└── DocGenius/
    └── PDFChat.py         # stripped-down single-PDF version, kept as reference
```

---

## License

GPL v3. See LICENSE for details.

---

## Credit

Original project by [KalyanMurapaka45](https://github.com/KalyanMurapaka45).
