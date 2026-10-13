import os
import io
import json
import hashlib
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
import google.genai as genai

# grab the API key from .env if it exists
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

st.set_page_config(
    page_title="DocGenius – AI PDF Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- styling ---
# went with a dark theme, kept it clean without too much going on
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 { color: #e8f4fd; font-size: 2.5rem; font-weight: 700; margin: 0; }
    .main-header p  { color: #a8c4d8; font-size: 1rem; margin: 0.5rem 0 0 0; }

    /* user messages get a blue left border, assistant gets green */
    .chat-bubble {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        animation: popIn 0.25s ease-in-out;
    }
    @keyframes popIn { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }

    .chat-bubble.user      { background: #1e3a5f; border-left: 4px solid #4a9eff; }
    .chat-bubble.assistant { background: #1a2744; border-left: 4px solid #00c896; }
    .chat-bubble .icon     { font-size: 1.4rem; flex-shrink: 0; margin-top: 2px; }
    .chat-bubble .text     { color: #e2eaf4; line-height: 1.6; flex: 1; }
    .chat-bubble .time     { font-size: 0.72rem; color: #6b8aad; margin-top: 4px; }

    /* little card that shows the source chunk preview */
    .source-card {
        background: #0d1b2a;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 0.6rem 0.9rem;
        margin: 0.3rem 0;
        font-size: 0.82rem;
        color: #a8c4d8;
    }
    .source-card strong { color: #4a9eff; }

    .stat-box {
        background: #1a2744;
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stat-box .num   { font-size: 1.8rem; font-weight: 700; color: #4a9eff; }
    .stat-box .label { font-size: 0.78rem; color: #6b8aad; margin-top: 2px; }

    .stButton > button { border-radius: 8px; font-weight: 500; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- helper functions ---

def get_file_hash(raw_bytes: bytes) -> str:
    # use first 10 chars of md5 just to identify if we've seen this file before
    return hashlib.md5(raw_bytes).hexdigest()[:10]


@st.cache_data(show_spinner=False)
def read_pdfs(file_bytes_list: list, file_names: list):
    """
    Loop through each uploaded PDF, pull out all the text,
    and keep track of which page each chunk of text came from.
    Returns the combined text and a list of page metadata dicts.
    """
    all_text = ""
    page_locations = []  # each entry maps a char range back to a file + page number

    for raw, name in zip(file_bytes_list, file_names):
        reader = PdfReader(io.BytesIO(raw))
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            start_idx = len(all_text)
            all_text += page_text + "\n"
            page_locations.append({
                "file": name,
                "page": page_num + 1,
                "start": start_idx,
                "end": len(all_text),
            })

    return all_text, page_locations


@st.cache_data(show_spinner=False)
def build_index(full_text: str, key: str):
    """
    Split the text into overlapping chunks and embed them into a FAISS index.
    Using RecursiveCharacterTextSplitter so it tries to split on paragraph breaks
    first rather than just chopping at a fixed character count.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(full_text)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=key,
    )

    # FAISS runs locally so no extra API calls for search
    index = FAISS.from_texts(chunks, embedding=embeddings)
    return index, chunks


def get_answer(question: str, index, key: str, history: list, temp: float = 0.3):
    """
    The main RAG function. Retrieves relevant chunks from the index,
    builds a prompt with the last few conversation turns for context,
    then calls Gemini and returns the answer + the source docs it used.
    """
    retriever = index.as_retriever(search_kwargs={"k": 5})
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=key,
        temperature=temp,
    )

    # only pass the last 6 turns so we don't blow up the context window
    recent_history = history[-6:]
    history_messages = []
    for user_msg, bot_msg in recent_history:
        history_messages.append(HumanMessage(content=user_msg))
        history_messages.append(AIMessage(content=bot_msg))

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are DocGenius, an AI assistant that answers questions about uploaded PDF documents.\n"
         "Only use the context below to answer. Be specific and reference details from the text.\n"
         "If you genuinely can't find the answer in the context, say so — don't make things up.\n\n"
         "Context:\n{context}"),
        *[(m.type, m.content) for m in history_messages],
        ("human", "{question}"),
    ])

    # pull the chunks first so we can show sources to the user
    matched_docs = retriever.invoke(question)
    context_str = "\n\n".join(d.page_content for d in matched_docs)

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context_str, "question": question})

    return answer, matched_docs


@st.cache_data(show_spinner=False)
def summarize_doc(text: str, key: str) -> str:
    """Quick summary of the document using Gemini directly (not through LangChain)."""
    client = genai.Client(api_key=key)

    # trim to 8k chars — enough for a good summary without hitting token limits
    excerpt = text[:8000]

    prompt = (
        "Summarize this document. Include:\n"
        "1. What it's about\n"
        "2. Key points as a bullet list\n"
        "3. Any important conclusions or findings\n\n"
        f"Document:\n{excerpt}"
    )

    resp = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
    return resp.text


def chat_to_txt(history: list) -> str:
    """Format the chat history as plain text for the export button."""
    lines = [
        "DocGenius Chat Export",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 50,
        "",
    ]
    for msg in history:
        who = "You" if msg["role"] == "user" else "DocGenius"
        lines.append(f"[{msg['time']}] {who}:")
        lines.append(msg["content"])
        lines.append("")
    return "\n".join(lines)


def chat_to_json(history: list) -> str:
    return json.dumps(history, indent=2, ensure_ascii=False)


# --- session state setup ---
# just making sure all the keys exist on first load

def init_session():
    defaults = {
        "messages": [],        # full chat log shown in the UI
        "qa_pairs": [],        # (question, answer) tuples fed back into the prompt as history
        "index": None,         # the FAISS vector store
        "doc_text": "",        # all extracted text across every uploaded PDF
        "page_map": [],        # maps char positions back to file + page
        "summary": "",         # auto-generated summary
        "seen_files": set(),   # hashes of files we've already processed
        "loaded_files": [],    # display names of loaded files
        "page_count": 0,
        "word_count": 0,
        "temp": 0.3,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# --- sidebar ---

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # let the user paste a key in the UI if they don't have a .env file
    key_input = st.text_input(
        "Google API Key",
        value=api_key or "",
        type="password",
        help="Get a free key at https://aistudio.google.com/app/apikey",
    )
    if key_input:
        api_key = key_input

    st.divider()

    st.markdown("## 📂 Upload Documents")
    uploads = st.file_uploader(
        "Drop your PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    process_btn = st.button("⚡ Process Documents", use_container_width=True, type="primary")

    st.divider()

    st.markdown("## 🎛️ Response Settings")
    st.session_state.temp = st.slider(
        "Creativity (temperature)",
        min_value=0.0, max_value=1.0,
        value=st.session_state.temp, step=0.05,
        help="Low = stick to the facts, High = more creative/loose",
    )

    st.divider()

    # show some stats once files are loaded
    if st.session_state.loaded_files:
        st.markdown("## 📊 Loaded Documents")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                f'<div class="stat-box"><div class="num">{len(st.session_state.loaded_files)}</div>'
                f'<div class="label">Files</div></div>', unsafe_allow_html=True
            )
        with col_b:
            st.markdown(
                f'<div class="stat-box"><div class="num">{st.session_state.page_count}</div>'
                f'<div class="label">Pages</div></div>', unsafe_allow_html=True
            )
        st.markdown(
            f'<div class="stat-box" style="margin-top:0.5rem">'
            f'<div class="num">{st.session_state.word_count:,}</div>'
            f'<div class="label">Words indexed</div></div>', unsafe_allow_html=True
        )
        st.caption(", ".join(st.session_state.loaded_files))

    st.divider()

    st.markdown("## 💾 Export & Reset")

    if st.session_state.messages:
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "📝 TXT",
                data=chat_to_txt(st.session_state.messages),
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with c2:
            st.download_button(
                "🗂 JSON",
                data=chat_to_json(st.session_state.messages),
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True,
            )

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.qa_pairs = []
        st.rerun()

    if st.button("🔄 Start Over", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# --- main page ---

st.markdown(
    '<div class="main-header">'
    "<h1>📄 DocGenius</h1>"
    "<p>Upload your PDFs, ask questions, get answers with sources</p>"
    "</div>",
    unsafe_allow_html=True,
)

# stop here if there's no API key — nothing works without it
if not api_key:
    st.warning(
        "🔑 No API key found. Add it in the sidebar or put `GOOGLE_API_KEY=...` in a `.env` file. "
        "Get a free key at [Google AI Studio](https://aistudio.google.com/app/apikey).",
        icon="⚠️",
    )
    st.stop()


# --- document processing ---

if process_btn and uploads:
    # filter out anything we've already processed this session
    new_uploads = []
    for f in uploads:
        raw = f.read()
        fhash = get_file_hash(raw)
        if fhash not in st.session_state.seen_files:
            new_uploads.append((f.name, raw, fhash))

    if not new_uploads:
        st.info("These files have already been processed.", icon="ℹ️")
    else:
        with st.spinner(f"Reading {len(new_uploads)} file(s)..."):
            raw_bytes = [b for _, b, _ in new_uploads]
            names = [n for n, _, _ in new_uploads]

            new_text, new_pages = read_pdfs([bytes(b) for b in raw_bytes], names)

            # merge with anything already loaded
            combined_text = st.session_state.doc_text + new_text
            combined_pages = st.session_state.page_map + new_pages

            new_index, _ = build_index(combined_text, api_key)

            # update session
            for name, _, fhash in new_uploads:
                st.session_state.seen_files.add(fhash)
                if name not in st.session_state.loaded_files:
                    st.session_state.loaded_files.append(name)

            st.session_state.doc_text = combined_text
            st.session_state.page_map = combined_pages
            st.session_state.index = new_index
            st.session_state.page_count = len(combined_pages)
            st.session_state.word_count = len(combined_text.split())

        with st.spinner("Summarizing..."):
            st.session_state.summary = summarize_doc(combined_text, api_key)

        st.success(f"Done — {len(new_uploads)} file(s) loaded, {st.session_state.page_count} pages indexed.")
        st.rerun()

elif process_btn and not uploads:
    st.warning("Upload at least one PDF first.", icon="⚠️")


# show the summary in a collapsible section
if st.session_state.summary:
    with st.expander("📋 Document Summary", expanded=False):
        st.markdown(st.session_state.summary)


# --- chat UI ---

if st.session_state.index:

    # render all messages so far
    for msg in st.session_state.messages:
        icon = "🧑" if msg["role"] == "user" else "🤖"
        role_class = msg["role"]
        st.markdown(
            f'<div class="chat-bubble {role_class}">'
            f'<span class="icon">{icon}</span>'
            f'<div><div class="text">{msg["content"]}</div>'
            f'<div class="time">{msg["time"]}</div></div>'
            f"</div>",
            unsafe_allow_html=True,
        )
        # show sources under each assistant reply
        if msg.get("sources"):
            with st.expander(f"📌 {len(msg['sources'])} source(s) used", expanded=False):
                for s in msg["sources"]:
                    st.markdown(
                        f'<div class="source-card"><strong>{s["file"]} · Page {s["page"]}</strong>'
                        f'<br>{s["preview"]}</div>',
                        unsafe_allow_html=True,
                    )

    st.markdown("---")

    # text input + send button in a form so Enter key works
    with st.form("question_form", clear_on_submit=True):
        q_col, btn_col = st.columns([5, 1])
        with q_col:
            user_question = st.text_input(
                "Ask something...",
                placeholder="e.g. What is the main argument? Summarize section 2.",
                label_visibility="collapsed",
            )
        with btn_col:
            send = st.form_submit_button("Send ➤", use_container_width=True, type="primary")

    # quick buttons for common questions
    st.markdown("**Try asking:**")
    btn_cols = st.columns(4)
    shortcuts = [
        "📝 Summarize this document",
        "🔑 What are the key points?",
        "📊 Any data or statistics?",
        "❓ What questions does this raise?",
    ]
    shortcut_clicked = None
    for i, label in enumerate(shortcuts):
        if btn_cols[i].button(label, use_container_width=True):
            # strip the emoji prefix before sending
            shortcut_clicked = label.split(" ", 1)[1]

    # figure out what question to actually run
    final_q = user_question if send and user_question else shortcut_clicked

    if final_q:
        now = datetime.now().strftime("%H:%M")

        # save the user's message
        st.session_state.messages.append({
            "role": "user",
            "content": final_q,
            "time": now,
        })

        with st.spinner("Thinking..."):
            try:
                answer, source_docs = get_answer(
                    final_q,
                    st.session_state.index,
                    api_key,
                    st.session_state.qa_pairs,
                    st.session_state.temp,
                )

                # figure out which file/page each source chunk came from
                sources = []
                for doc in source_docs:
                    chunk = doc.page_content
                    # find where this chunk sits in the full text
                    pos = st.session_state.doc_text.find(chunk[:80])
                    location = {"file": "Unknown", "page": "?"}
                    if pos >= 0:
                        for entry in st.session_state.page_map:
                            if entry["start"] <= pos < entry["end"]:
                                location = entry
                                break
                    sources.append({
                        "file": location["file"],
                        "page": location["page"],
                        "preview": chunk[:200].replace("\n", " ") + "…",
                    })

                # save to both history stores
                st.session_state.qa_pairs.append((final_q, answer))
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "time": datetime.now().strftime("%H:%M"),
                    "sources": sources,
                })

            except Exception as err:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Something went wrong: {str(err)}",
                    "time": datetime.now().strftime("%H:%M"),
                })

        st.rerun()

else:
    # nothing loaded yet — show a welcome screen
    st.markdown(
        """
        <div style='text-align:center; padding:3rem 1rem; color:#6b8aad;'>
            <div style='font-size:4rem;'>📂</div>
            <h3 style='color:#a8c4d8; margin:0.5rem 0;'>No documents loaded</h3>
            <p>Upload a PDF in the sidebar and hit <strong>⚡ Process Documents</strong> to get started.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # feature overview cards
    with st.expander("What can this do?", expanded=True):
        c1, c2, c3 = st.columns(3)
        features = [
            ("📚", "Multiple PDFs", "Load several docs and query across all of them"),
            ("🧠", "Remembers context", "Follow-up questions work naturally"),
            ("📋", "Auto summary", "Get a summary of any doc right after upload"),
            ("📌", "Source citations", "See exactly which page each answer came from"),
            ("💾", "Export chat", "Save your Q&A as a .txt or .json file"),
            ("🎛️", "Adjustable AI", "Tune how creative or factual the responses are"),
        ]
        for i, (icon, title, desc) in enumerate(features):
            with [c1, c2, c3][i % 3]:
                st.markdown(
                    f"<div style='background:#1a2744;border-radius:10px;padding:1rem;"
                    f"margin:0.3rem 0;text-align:center'>"
                    f"<div style='font-size:2rem'>{icon}</div>"
                    f"<strong style='color:#e2eaf4'>{title}</strong>"
                    f"<p style='color:#6b8aad;font-size:0.85rem;margin:0.3rem 0 0 0'>{desc}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

# footer
st.markdown(
    "<hr style='margin-top:2rem;border-color:#1e3a5f'>"
    "<p style='text-align:center;color:#3d5a7a;font-size:0.78rem'>"
    "DocGenius · Google Gemini · LangChain · Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
