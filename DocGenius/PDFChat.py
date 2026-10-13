"""
DocGenius – PDFChat.py
Legacy single-file entry point (kept for reference).
The full-featured app is in /app.py at the project root.
"""

import os
import io
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="DocGenius – Ask Your PDF", page_icon="📄")
st.header("📄 Ask Your PDF")
st.caption("Single-document mode · For the full experience, run app.py")

pdf = st.file_uploader("Upload a PDF", type="pdf")

if pdf is not None:
    reader = PdfReader(io.BytesIO(pdf.read()))
    text = "".join(page.extract_text() or "" for page in reader.pages)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_text(text)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY,
    )
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    query = st.text_input("Ask a question about your PDF")
    if query:
        docs = knowledge_base.similarity_search(query, k=4)

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3,
            convert_system_message_to_human=True,
        )

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "Use the context below to answer the question. "
                "If the answer is not in the context, say so.\n\n"
                "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            ),
        )

        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
        response = chain.run(input_documents=docs, question=query)
        st.success(response)
