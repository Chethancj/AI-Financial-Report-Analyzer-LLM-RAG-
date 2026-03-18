import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np

st.title("AI Financial Report Analyzer")

uploaded_file = st.file_uploader("Upload Financial Report PDF")

if uploaded_file:

    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        text += page.extract_text()

    def split_text(text, chunk_size=500):
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    chunks = split_text(text)

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(chunks)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

    def search(query, k=3):
        query_embedding = embedder.encode([query])
        D, I = index.search(query_embedding, k)
        return [chunks[i] for i in I[0]]

    def ask_question(query):
        context = " ".join(search(query))
        prompt = f"Answer based on context:\n{context}\n\nQuestion: {query}"
        result = qa_pipeline(prompt, max_length=200)
        return result[0]['generated_text']

    question = st.text_input("Ask a question")

    if question:
        answer = ask_question(question)
        st.write(answer)
