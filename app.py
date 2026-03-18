import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

# =========================
# CONFIG (UPDATED ENDPOINT)
# =========================
API_URL = "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.2"

headers = {
    "Authorization": f"Bearer {st.secrets['HUGGINGFACE_API_KEY']}",
    "Content-Type": "application/json"
}

st.title("📊 AI Financial Report Analyzer (Hugging Face)")

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
    # SMART CHUNKING (OVERLAP)
    # =========================
    def split_text(text, chunk_size=1000, overlap=200):
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i+chunk_size])
        return chunks

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
    def search(query, k=6):
        query_embedding = embedder.encode([query])
        D, I = index.search(query_embedding, k)
        return [chunks[i] for i in I[0]]

    # =========================
    # HUGGING FACE QA FUNCTION
    # =========================
    def ask_question(query):

        relevant_chunks = search(query, k=6)
        context = "\n\n".join(relevant_chunks)

        prompt = f"""
You are a financial analyst AI.

Answer clearly and professionally using ONLY the context below.

Rules:
- Present numbers cleanly
- Use bullet points or tables when helpful
- If answer is not found, say "Not found"

Context:
{context}

Question:
{query}

Answer:
"""

        try:
            response = requests.post(
                API_URL,
                headers=headers,
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 300,
                        "temperature": 0.2,
                        "return_full_text": False
                    }
                },
                timeout=60
            )

            # =========================
            # ERROR HANDLING
            # =========================
            if response.status_code != 200:
                return f"❌ API Error {response.status_code}: {response.text}"

            result = response.json()

            if isinstance(result, list):
                return result[0]["generated_text"].strip()

            elif "error" in result:
                return f"❌ Model Error: {result['error']}"

            else:
                return str(result)

        except requests.exceptions.Timeout:
            return "⏳ Request timed out. Try again or reduce query size."

        except Exception as e:
            return f"❌ Unexpected error: {e}"

    # =========================
    # USER INPUT
    # =========================
    question = st.text_input("Ask a question about the report")

    if question:
        with st.spinner("Analyzing document... ⏳"):
            answer = ask_question(question)
            st.success(answer)
