import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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
    # LOAD MODEL
    # =========================
    model_name = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # =========================
    # SEARCH FUNCTION
    # =========================
    def search(query, k=3):
        query_embedding = embedder.encode([query])
        D, I = index.search(query_embedding, k)
        return [chunks[i] for i in I[0]]

    # =========================
    # QA FUNCTION
    # =========================
    def ask_question(query):

        context = " ".join(search(query))

        prompt = f"""
        You are a financial analyst.

        Answer clearly based on the context below.

        Context:
        {context}

        Question:
        {query}

        Answer:
        """

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7
        )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if not answer.strip():
            return "No answer found. Try another question."

        return answer

    # =========================
    # QUESTION INPUT
    # =========================
    question = st.text_input("Ask a question")

    if question:
        with st.spinner("Analyzing document..."):
            answer = ask_question(question)
            st.success(answer)
