import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

# =========================
# GROQ CONFIG
# =========================
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

st.title("🧠 Autonomous Financial Analyst (Multi-Agent LLM - Groq)")

# =========================
# LLM CALL FUNCTION (GROQ)
# =========================
def call_llm(prompt, max_tokens=500):

    headers = {
        "Authorization": f"Bearer {st.secrets['GROQ_API_KEY']}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are a professional financial analyst AI."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(GROQ_URL, headers=headers, json=data, timeout=30)

        if response.status_code != 200:
            return f"❌ API Error {response.status_code}: {response.text}"

        result = response.json()
        return result["choices"][0]["message"]["content"].strip()

    except Exception as e:
        return f"❌ Error: {e}"

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload Financial Report PDF")

# =========================
# MAIN PIPELINE
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
    # CHUNKING
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
    # RETRIEVAL
    # =========================
    def retrieve(query, k=6):
        query_embedding = embedder.encode([query])
        D, I = index.search(query_embedding, k)
        return [chunks[i] for i in I[0]]

    # =========================
    # AGENTS
    # =========================
    def financial_agent(context):
        prompt = f"""
Extract key financial metrics:
- Revenue
- Profit
- Costs
- Growth trends

Context:
{context}
"""
        return call_llm(prompt)


    def risk_agent(context):
        prompt = f"""
Identify:
- Financial risks
- Operational risks
- Market risks

Context:
{context}
"""
        return call_llm(prompt)


    def strategy_agent(context):
        prompt = f"""
Summarize:
- Business strategy
- Future outlook
- Competitive positioning

Context:
{context}
"""
        return call_llm(prompt)

    # =========================
    # REASONING LOOP
    # =========================
    def reasoning_improve(answer):

        critique = call_llm(f"""
Critique the following answer for:
- Accuracy
- Missing insights
- Clarity

Answer:
{answer}
""")

        improved = call_llm(f"""
Improve the answer using this critique.

Answer:
{answer}

Critique:
{critique}

Return a well-structured response with:
📊 Financial Summary
⚠️ Risks
🧠 Strategy
""")

        return improved

    # =========================
    # ORCHESTRATOR
    # =========================
    def multi_agent_system(query):

        relevant_chunks = retrieve(query, k=6)
        context = "\n\n".join(relevant_chunks)

        # Run agents
        financials = financial_agent(context)
        risks = risk_agent(context)
        strategy = strategy_agent(context)

        # Combine outputs
        combined_prompt = f"""
Create a structured financial report.

Financials:
{financials}

Risks:
{risks}

Strategy:
{strategy}

User Question:
{query}

Format clearly with headings.
"""
        draft = call_llm(combined_prompt)

        # Improve via reasoning loop
        final_answer = reasoning_improve(draft)

        return final_answer

    # =========================
    # USER INPUT
    # =========================
    question = st.text_input("Ask a financial question")

    if question:
        with st.spinner("Running multi-agent analysis... ⏳"):
            answer = multi_agent_system(question)
            st.success(answer)
