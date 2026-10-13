"""
DocGenius – PDFChat.py
Lightweight single-PDF version, kept as a reference.
The full app with multi-PDF, chat memory, and source citations is in /app.py
"""

import os
import io
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="DocGenius – Ask Your PDF", page_icon="📄")
st.header("📄 Ask Your PDF")
st.caption("Single-document mode · For the full experience run app.py")

pdf = st.file_uploader("Upload a PDF", type="pdf")

if pdf is not None:
    # pull all the text out page by page
    reader = PdfReader(io.BytesIO(pdf.read()))
    text = "".join(page.extract_text() or "" for page in reader.pages)

    # chunk it up so it fits in the embedding model
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_text(text)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    index = FAISS.from_texts(chunks, embeddings)

    query = st.text_input("Ask a question about your PDF")
    if query:
        # grab the top 4 most relevant chunks
        docs = index.similarity_search(query, k=4)
        context = "\n\n".join(d.page_content for d in docs)

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.3,
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Answer the question using only the context below. "
             "If the answer isn't there, say so.\n\nContext:\n{context}"),
            ("human", "{question}"),
        ])

        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": query})
        st.success(answer)
