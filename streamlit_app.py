import io
import json
import time
import zipfile
from typing import Optional, List, Dict

import pandas as pd
import streamlit as st

# Optional OpenAI SDK
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# -----------------------
# State
# -----------------------
def init_state():
    if "chat" not in st.session_state:
        st.session_state.chat = []  # [{"role":"user|assistant","content":"..."}]
    if "llm_confirmed" not in st.session_state:
        st.session_state.llm_confirmed = False
    if "last_confirm_result" not in st.session_state:
        st.session_state.last_confirm_result = ""
    if "datasets" not in st.session_state:
        # dict: filename -> dataframe
        st.session_state.datasets = {}


# -----------------------
# Defaults (Structured Prompts)
# -----------------------
DEFAULT_TASK = "You are an ODI mountain bike grips expert who provides grip recommendations to users."
DEFAULT_PERSONA = "The user is an experienced mountain biker. Use technical terms and slang."
DEFAULT_TONE = "Respond in a professional and informative tone, similar to a customer service representative."


def build_system_prompt(task: str, persona: str, tone: str, dataset_context: str) -> str:
    return f"""
You are an ODI mountain bike grips assistant.

[Task Definition]
{task}

[Customer Persona]
{persona}

[Tone & Language Style]
{tone}

[Dataset Context]
{dataset_context}

[Rules]
- Recommend ODI grips and explain why in MTB terms (feel, damping, tack, flange, diameter, terrain).
- If you need to clarify, ask at most 1–2 quick questions (hand size/glove size, terrain, flange preference, diameter feel).
- Prefer dataset facts when available; do not invent specs not present.
- Output format:
  1) Top pick (why)
  2) Runner-up (why)
  3) If rider preference differs (alternate)
""".strip()


def call_llm(api_key: str, model: str, system_prompt: str, messages: list, temperature: float) -> str:
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Add it to requirements.txt or replace call_llm().")

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "system", "content": system_prompt}] + messages,
    )
    return resp.choices[0].message.content


# -----------------------
# Dataset handling
# -----------------------
def load_csv_bytes(name: str, b: bytes) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(io.BytesIO(b))
    except Exception:
        return None


def load_zip_of_csvs(zip_bytes: bytes) -> Dict[str, pd.DataFrame]:
    out = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        for file_name in z.namelist():
            # Only load CSV files
            if file_name.lower().endswith(".csv"):
                data = z.read(file_name)
                df = load_csv_bytes(file_name, data)
                if df is not None:
                    out[file_name] = df
    return out


def build_dataset_context(datasets: Dict[str, pd.DataFrame], max_rows_each: int = 6) -> str:
    if not datasets:
        return "No dataset uploaded."

    lines = []
    lines.append(f"{len(datasets)} dataset file(s) loaded.")

    for fname, df in list(datasets.items())[:12]:  # cap how many we summarize
        lines.append(f"\n--- File: {fname}")
        lines.append(f"Columns: {list(df.columns)}")
        lines.append("Preview:")
        lines.append(df.head(max_rows_each).to_csv(index=False))

    if len(datasets) > 12:
        lines.append(f"\n(And {len(datasets) - 12} more files not shown here.)")

    return "\n".join(lines)


def transcript_json() -> str:
    return json.dumps(
        {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "messages": st.session_state.chat,
        },
        indent=2,
    )


# =========================
# App
# =========================
init_state()

st.set_page_config(page_title="ODI Grips Chatbot", page_icon="🚵", layout="wide")
st.title("🚵 ODI Grips Chatbot")
st.caption("Structured prompts + dataset upload + transcript download")


# ---- LLM settings (keep in sidebar, but buttons moved to main page) ----
with st.sidebar:
    st.header("LLM Settings")
    api_key = st.text_input("API Key", value=st.secrets.get("OPENAI_API_KEY", ""), type="password")
    model = st.text_input("Model", value=st.secrets.get("OPENAI_MODEL", "gpt-4o-mini"))
    temperature = st.slider("Temperature", 0.0, 1.0, 0.35, 0.05)

    if st.session_state.last_confirm_result:
        st.caption(st.session_state.last_confirm_result)
        st.success("LLM confirmed ✅" if st.session_state.llm_confirmed else "LLM not confirmed ⚠️")


# ---- Structured Prompts FIRST ----
st.subheader("Structured Prompts")
with st.expander("Structured Prompts", expanded=True):
    task = st.text_area("Task Definition", value=DEFAULT_TASK, height=90)
    persona = st.text_area("Customer Persona", value=DEFAULT_PERSONA, height=90)
    tone = st.text_area("Tone & Language Style", value=DEFAULT_TONE, height=90)

# ---- Dataset Upload SECOND ----
st.subheader("Dataset Upload (ODI products)")
st.write("Upload either **a ZIP of your folder** (recommended) or **multiple CSV files**.")

zip_file = st.file_uploader("Upload ZIP (contains CSVs)", type=["zip"])
csv_files = st.file_uploader("Or upload multiple CSVs", type=["csv"], accept_multiple_files=True)

load_col1, load_col2 = st.columns([1, 1])

with load_col1:
    if st.button("Load Uploaded Data", use_container_width=True):
        loaded = {}

        if zip_file is not None:
            try:
                loaded = load_zip_of_csvs(zip_file.read())
                st.session_state.datasets.update(loaded)
                st.success(f"Loaded {len(loaded)} CSV(s) from ZIP.")
            except Exception as e:
                st.error(f"ZIP load failed: {e}")

        if csv_files:
            count = 0
            for f in csv_files:
                df = load_csv_bytes(f.name, f.read())
                if df is not None:
                    st.session_state.datasets[f.name] = df
                    count += 1
            st.success(f"Loaded {count} CSV(s) from manual upload.")

with load_col2:
    if st.button("Clear Loaded Data", use_container_width=True):
        st.session_state.datasets = {}
        st.toast("Datasets cleared.")


if st.session_state.datasets:
    st.markdown("### Loaded files")
    st.write(list(st.session_state.datasets.keys())[:30])
    # Show preview selector
    preview_file = st.selectbox("Preview a file", options=list(st.session_state.datasets.keys()))
    st.dataframe(st.session_state.datasets[preview_file].head(25), use_container_width=True)
else:
    st.info("No datasets loaded yet.")


# ---- Buttons moved BELOW prompts (and dataset) THIRD ----
st.subheader("Actions")
b1, b2, b3 = st.columns(3)

with b1:
    if st.button("✅ Confirm LLM Setup", use_container_width=True):
        if not api_key:
            st.session_state.llm_confirmed = False
            st.session_state.last_confirm_result = "Missing API key."
        else:
            try:
                ping_system = "You are a helpful assistant."
                ping_messages = [{"role": "user", "content": "Reply with: LLM OK"}]
                out = call_llm(api_key, model, ping_system, ping_messages, temperature=0.0)
                st.session_state.llm_confirmed = "LLM OK" in out
                st.session_state.last_confirm_result = f"Response: {out}"
                st.toast("LLM setup checked.")
            except Exception as e:
                st.session_state.llm_confirmed = False
                st.session_state.last_confirm_result = f"Error: {e}"

with b2:
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.chat = []
        st.toast("Chat cleared.")

with b3:
    st.download_button(
        "⬇️ Generate Transcript",
        data=transcript_json().encode("utf-8"),
        file_name="odi_chat_transcript.json",
        mime="application/json",
        use_container_width=True,
    )


st.divider()

# ---- Chat LAST (organized) ----
st.subheader("Chat")

# show history
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_msg = st.chat_input("Ask: Which ODI grips would you recommend?")

if user_msg:
    st.session_state.chat.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    dataset_context = build_dataset_context(st.session_state.datasets)

    system_prompt = build_system_prompt(task, persona, tone, dataset_context)

    with st.chat_message("assistant"):
        if not api_key:
            st.error("Add your API key in the sidebar first.")
        else:
            try:
                assistant_text = call_llm(
                    api_key=api_key,
                    model=model,
                    system_prompt=system_prompt,
                    messages=st.session_state.chat,
                    temperature=temperature,
                )
                st.markdown(assistant_text)
                st.session_state.chat.append({"role": "assistant", "content": assistant_text})
            except Exception as e:
                st.error(f"LLM error: {e}")
