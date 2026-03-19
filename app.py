import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

# =========================
# CONFIG
# =========================
API_URL = "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.2"

headers = {
    "Authorization": f"Bearer {st.secrets['HUGGINGFACE_API_KEY']}",
    "Content-Type": "application/json"
}

st.title("🧠 Autonomous Financial Analyst (Multi-Agent LLM)")

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload Financial Report PDF")

# =========================
# LLM CALL FUNCTION
# =========================
def call_llm(prompt, max_tokens=300):
    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": 0.2,
                    "return_full_text": False
                }
            },
            timeout=60
        )

        if response.status_code != 200:
            return f"❌ API Error {response.status_code}: {response.text}"

        result = response.json()

        if isinstance(result, list):
            return result[0]["generated_text"].strip()
        elif "error" in result:
            return f"❌ Model Error: {result['error']}"
        else:
            return str(result)

    except Exception as e:
        return f"❌ Error: {e}"

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
    # SEARCH
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
You are a financial analyst.

Extract key financial metrics:
- Revenue
- Profit
- Costs
- Growth trends

Context:
{context}

Return clean bullet points.
"""
        return call_llm(prompt)


    def risk_agent(context):
        prompt = f"""
You are a risk analyst.

Identify:
- Financial risks
- Operational risks
- Market risks

Context:
{context}

Return concise bullet points.
"""
        return call_llm(prompt)


    def strategy_agent(context):
        prompt = f"""
You are a strategy analyst.

Summarize:
- Business strategy
- Future outlook
- Competitive positioning

Context:
{context}

Return concise insights.
"""
        return call_llm(prompt)

    # =========================
    # REASONING LOOP
    # =========================
    def reasoning_improve(answer):
        critique = call_llm(f"Critique this answer for accuracy and completeness:\n{answer}")

        improved = call_llm(f"""
Improve the answer using this critique.

Answer:
{answer}

Critique:
{critique}

Return a refined, structured response.
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
You are a senior financial analyst.

Create a structured report using:

Financials:
{financials}

Risks:
{risks}

Strategy:
{strategy}

User Question:
{query}

Format:
📊 Financial Summary
⚠️ Risks
🧠 Strategy
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
