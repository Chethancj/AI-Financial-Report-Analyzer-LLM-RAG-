import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# =========================
# TITLE
# =========================
st.title("AI Financial Report Analyzer")

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload Financial Report PDF")

# =========================
# MAIN LOGIC
# =========================
if uploaded_file:

    st.success("File uploaded successfully!")

    # =========================
    # READ PDF
    # =========================
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    if not text.strip():
        st.error("No readable text found in PDF")
        st.stop()

    # =========================
    # SPLIT TEXT
    # =========================
    def split_text(text, chunk_size=500):
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    chunks = split_text(text)

    st.write("Chunks created:", len(chunks))

    # =========================
    # EMBEDDINGS
    # =========================
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(chunks)

    # =========================
    # FAISS INDEX
    # =========================
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    # =========================
    # SEARCH FUNCTION
    # =========================
    def search(query, k=3):
        query_embedding = embedder.encode([query])
        D, I = index.search(query_embedding, k)
        return [chunks[i] for i in I[0]]

    # =========================
    # STABLE QA FUNCTION (FIXED)
    # =========================
    def ask_question(query):

        relevant_chunks = search(query)

        context = " ".join(relevant_chunks)  # keep small for stability

        if "revenue" in query.lower():
            return context[:2000]

        elif "summary" in query.lower():
            return context[:2000]

        else:
            return "Relevant information:\n\n" + context[:500]

    # =========================
    # QUESTION INPUT
    # =========================
    question = st.text_input("Ask a question")

    if question:
        with st.spinner("Analyzing document..."):
            answer = ask_question(question)
            st.success(answer)
