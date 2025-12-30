# Full Streamlit App with RAG (Retrieval-Augmented Generation) for ODI Grip Chatbot

import json
import time
from typing import Optional, Dict, List, Any

import pandas as pd
import requests
import streamlit as st

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# ------------------
# Initialize State
# ------------------
def init_state():
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "datasets" not in st.session_state:
        st.session_state.datasets = {}
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None
    if "embed_model" not in st.session_state:
        st.session_state.embed_model = SentenceTransformer("all-MiniLM-L6-v2")


# ------------------
# Vector DB Functions
# ------------------
def init_vector_db():
    client = chromadb.Client()
    collection = client.create_collection("odi_grips")
    st.session_state.vector_db = collection


def embed_texts(texts: List[str]) -> List[List[float]]:
    return st.session_state.embed_model.encode(texts).tolist()


def add_csv_to_vector_db(datasets: Dict[str, pd.DataFrame]):
    if st.session_state.vector_db is None:
        init_vector_db()

    all_texts, metadatas, ids = [], [], []
    chunk_id = 0
    for fname, df in datasets.items():
        for _, row in df.iterrows():
            text = ", ".join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])])
            all_texts.append(text)
            metadatas.append({"source_file": fname})
            ids.append(f"chunk-{chunk_id}")
            chunk_id += 1

    embeddings = embed_texts(all_texts)
    st.session_state.vector_db.add(documents=all_texts, embeddings=embeddings, metadatas=metadatas, ids=ids)


def rag_retrieve_context(query: str, top_k: int = 5) -> str:
    if st.session_state.vector_db is None:
        return "No product knowledge loaded."
    query_vec = embed_texts([query])[0]
    results = st.session_state.vector_db.query(query_embeddings=[query_vec], n_results=top_k)
    return "\n\n".join(results["documents"][0]) if results["documents"] else "No relevant info found."


# ------------------
# Prompt Builder
# ------------------
def build_system_prompt(task, persona, tone, data_rules, scope, pref_schema, mapping_guide, workflow, output_rules, rag_context):
    return f"""
[Task Definition]
{task}

[Customer Persona]
{persona}

[Tone & Language Style]
{tone}

[Data Access & Grounding Rules]
{data_rules}

[Scope & Category Handling]
{scope}

[Preference Schema]
{pref_schema}

[Mapping Guide]
{mapping_guide}

[Conversation Workflow]
{workflow}

[Output Format Rules]
{output_rules}

[RAG Context Retrieved for this Query]
{rag_context}

IMPORTANT:
- Use ONLY the context above. Do NOT invent.
- If the context lacks details, ask a follow-up question.
""".strip()


# ------------------
# LLM Call
# ------------------
def call_llm_openrouter(api_key: str, model: str, system_prompt: str, messages: List[Dict[str, str]]) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    payload = {
        "model": model,
        "temperature": 0.2,
        "max_tokens": 600,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "ODI Grips Chatbot (RAG)",
    }
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ------------------
# Streamlit UI
# ------------------
init_state()

st.set_page_config(page_title="ODI Grips Chatbot with RAG", layout="wide")
st.title("🚵 ODI Grips Chatbot with RAG")

# Sidebar config
with st.sidebar:
    st.header("LLM Settings")
    api_key = st.text_input("OpenRouter API Key", type="password")
    model = st.text_input("Model", value="openai/gpt-4o-mini")

# Prompt controls
st.subheader("Prompt Controls")
task = st.text_area("Task", value="You are an ODI grip expert helping riders choose the right grips.")
persona = st.text_area("Persona", value="The user is a mountain biker who wants comfort and control.")
tone = st.text_area("Tone", value="Friendly, knowledgeable, and brief.")
data_rules = st.text_area("Data Rules", value="Use only product info retrieved. Never invent.")
scope = st.text_area("Scope", value="All ODI grips across all riding categories.")
pref_schema = st.text_area("Preference Schema", value="riding_style, thickness, locking_mechanism, damping")
mapping_guide = st.text_area("Mapping Guide", value="‘large hands’ = ‘thick’ grip")
workflow = st.text_area("Workflow", value="1. Ask riding type. 2. Identify needs. 3. Recommend grip.")
output_rules = st.text_area("Output Rules", value="Keep replies under 6 sentences. Use plain English.")

# Upload and Embed
st.subheader("📁 Upload ODI Grip CSVs")
csv_files = st.file_uploader("Upload CSVs", type=["csv"], accept_multiple_files=True)
if st.button("🔄 Load and Embed CSVs"):
    st.session_state.datasets = {}
    for f in csv_files:
        df = pd.read_csv(f)
        st.session_state.datasets[f.name] = df
    add_csv_to_vector_db(st.session_state.datasets)
    st.success("Data loaded and embedded into RAG database.")

# Chat
st.subheader("💬 Chat")
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_msg = st.chat_input("Ask a grip question...")
if user_msg:
    st.session_state.chat.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    rag_context = rag_retrieve_context(user_msg)
    full_prompt = build_system_prompt(task, persona, tone, data_rules, scope, pref_schema, mapping_guide, workflow, output_rules, rag_context)

    with st.chat_message("assistant"):
        if not api_key:
            st.error("Please enter your API key.")
        else:
            try:
                reply = call_llm_openrouter(api_key, model, full_prompt, st.session_state.chat)
                st.markdown(reply)
                st.session_state.chat.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error(f"LLM error: {e}")
