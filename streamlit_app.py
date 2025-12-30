import json
import urllib.parse
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# ----------------------------
# Constants + helpers
# ----------------------------
ALLOWED_PREFS = {
    "riding_style": ["trail", "enduro", "downhill", "cross-country"],
    "locking_mechanism": ["lock-on", "slip-on"],
    "thickness": ["thin", "medium", "thick", "medium-thick size xl"],
    "damping_level": ["low", "medium", "high"],
    "durability": ["low", "medium", "high"],
}

PREF_KEYS = list(ALLOWED_PREFS.keys())

PREMIUM_MODEL = "openai/gpt-4o-mini"
BUDGET_MODEL = "mistralai/mistral-small-latest"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

DEMO_RESPONSES = [
    "Appreciate the details! Based on that, I'm thinking about comfort and control--can you tell me how rough your usual trails are?",
    "Totally get it. If your hands go numb, cushioning matters. Would you lean toward softer rubber or extra support?",
    "Great intel. I'm picturing a good all-rounder, but help me narrow it down: lock-on hardware or more of a slip-on feel?",
    "Thanks! Once I know how thick you like the grips or how durable they need to be, I can zero in on a couple of winners.",
]


def odi_search_link(product_name: str) -> str:
    q = urllib.parse.quote(product_name or "")
    return f"https://odigrips.com/search?q={q}"


def normalize_listish(cell):
    if pd.isna(cell):
        return []
    return [x.strip().lower() for x in str(cell).split(";") if x.strip()]


def score_row(row, prefs):
    score = 0

    if prefs.get("locking_mechanism"):
        if str(row.get("locking_mechanism", "")).strip().lower() == prefs["locking_mechanism"]:
            score += 4
        else:
            score -= 2

    if prefs.get("riding_style"):
        styles = normalize_listish(row.get("riding_style"))
        if prefs["riding_style"] in styles:
            score += 4

    if prefs.get("thickness"):
        if str(row.get("thickness", "")).strip().lower() == prefs["thickness"]:
            score += 2

    if prefs.get("damping_level"):
        if str(row.get("damping_level", "")).strip().lower() == prefs["damping_level"]:
            score += 2

    if prefs.get("durability"):
        if str(row.get("durability", "")).strip().lower() == prefs["durability"]:
            score += 1

    return score


def recommend(df, prefs, top_n=3):
    scored = df.copy()
    scored["__score"] = scored.apply(lambda r: score_row(r, prefs), axis=1)
    scored = scored.sort_values("__score", ascending=False)
    return scored.head(top_n).drop(columns="__score")


def load_default_dataframe():
    """
    Tries:
      1) repo_root/data/ODI_MTB_GRIPS.csv
      2) repo_root/ODI_MTB_GRIPS.csv
    """
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "data" / "ODI_MTB_GRIPS.csv",
        repo_root / "ODI_MTB_GRIPS.csv",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p), p
    raise FileNotFoundError(
        "Could not find ODI_MTB_GRIPS.csv.\n"
        f"Looked for:\n- {candidates[0]}\n- {candidates[1]}\n\n"
        "Fix: add the CSV to your repo at one of those paths (recommended: data/ODI_MTB_GRIPS.csv)."
    )


# ----------------------------
# Session state helpers
# ----------------------------
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hey there! I'm Emily from the ODI bike shop. Tell me about your riding and I'll help you dial in the right grips.",
            }
        ]
    if "prefs" not in st.session_state:
        st.session_state.prefs = {k: "" for k in PREF_KEYS}
    if "demo_mode_forced" not in st.session_state:
        st.session_state.demo_mode_forced = False
    if "demo_error" not in st.session_state:
        st.session_state.demo_error = ""
    if "demo_resp_idx" not in st.session_state:
        st.session_state.demo_resp_idx = 0
    if "demo_notice_shown" not in st.session_state:
        st.session_state.demo_notice_shown = False
    if "low_cost_mode" not in st.session_state:
        st.session_state.low_cost_mode = False
    if "demo_mode_manual" not in st.session_state:
        st.session_state.demo_mode_manual = False


def reset_chat():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Fresh start! Share anything about your rides, hands, or grip wishes and I'll guide you.",
        }
    ]
    st.session_state.prefs = {k: "" for k in PREF_KEYS}
    st.session_state.demo_mode_forced = False
    st.session_state.demo_error = ""
    st.session_state.demo_resp_idx = 0
    st.session_state.demo_notice_shown = False


def enable_forced_demo(error_message: str):
    st.session_state.demo_mode_forced = True
    st.session_state.demo_error = error_message
    st.session_state.demo_notice_shown = False
    st.warning("OpenRouter hit a snag, so I'm switching to Demo mode for now. Let's keep chatting!")


def validate_pref_value(key: str, value: str) -> str:
    if not value:
        return ""
    normalized = value.strip().lower()
    if key == "riding_style" and normalized in {"xc", "x-c", "cross country", "crosscountry"}:
        normalized = "cross-country"
    if normalized in ALLOWED_PREFS[key]:
        return normalized
    return ""


def update_prefs(new_values: dict):
    for key in PREF_KEYS:
        val = new_values.get(key, "")
        valid = validate_pref_value(key, val)
        if valid:
            st.session_state.prefs[key] = valid


# ----------------------------
# Preference extraction
# ----------------------------
def call_openrouter(messages, model, max_tokens):
    api_key = st.secrets.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY in Streamlit secrets.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.6,
        "max_tokens": max_tokens,
    }
    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=25)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def llm_extract_preferences(user_text: str, model: str, max_tokens: int) -> dict:
    instructions = (
        "You extract mountain bike grip preferences. "
        "Return strict JSON ONLY with the keys riding_style, locking_mechanism, thickness, damping_level, durability. "
        "Each value must be one of the allowed options or an empty string. "
        "Allowed values:\n"
        f"riding_style: {', '.join(ALLOWED_PREFS['riding_style'])}\n"
        f"locking_mechanism: {', '.join(ALLOWED_PREFS['locking_mechanism'])}\n"
        f"thickness: {', '.join(ALLOWED_PREFS['thickness'])}\n"
        f"damping_level: {', '.join(ALLOWED_PREFS['damping_level'])}\n"
        f"durability: {', '.join(ALLOWED_PREFS['durability'])}\n"
        "If unsure, use an empty string."
    )
    payload = {
        "existing_preferences": st.session_state.prefs,
        "user_message": user_text,
    }
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": json.dumps(payload)},
    ]
    raw = call_openrouter(messages, model, max_tokens)
    return json.loads(raw)


def heuristic_prefs(user_text: str) -> dict:
    text = user_text.lower()
    found = {k: "" for k in PREF_KEYS}
    synonyms = {
        "riding_style": {
            "rocky": "trail",
            "all-mountain": "trail",
            "xc": "cross-country",
            "cross country": "cross-country",
        },
        "thickness": {
            "chunky": "thick",
            "fat": "thick",
            "slim": "thin",
            "skinny": "thin",
        },
        "damping_level": {
            "plush": "high",
            "cushy": "high",
            "firm": "low",
        },
        "durability": {
            "long lasting": "high",
            "long-lasting": "high",
        },
    }

    for key, options in ALLOWED_PREFS.items():
        for option in options:
            if option in text:
                found[key] = option
        for hint, canonical in synonyms.get(key, {}).items():
            if hint in text:
                found[key] = canonical

    if "lock on" in text or "lock-on" in text:
        found["locking_mechanism"] = "lock-on"
    if "slip on" in text or "slip-on" in text:
        found["locking_mechanism"] = "slip-on"

    return found


# ----------------------------
# Response generation
# ----------------------------
def build_system_prompt(low_cost: bool) -> str:
    limit_note = "Keep replies under about 80 words. " if low_cost else ""
    return (
        "You are an ODI bike shop employee, a warm, knowledgeable employee helping a rider pick ODI MTB grips. "
        "Listen carefully, acknowledge concerns, and keep things conversational. "
        "Ask one focused follow-up when information is missing. "
        "Offer simple tips drawn from common bike-fit wisdom, but leave the actual product scoring to the app. "
        f"{limit_note}"
        "If the rider is unsure, suggest easy ways to decide (feel preference, trail type, numbness, etc.)."
    )


def llm_chat_response(model: str, max_tokens: int):
    messages = [{"role": "system", "content": build_system_prompt(st.session_state.low_cost_mode)}]
    messages.extend(st.session_state.messages)
    return call_openrouter(messages, model, max_tokens)


def scripted_demo_response(user_text: str) -> str:
    idx = st.session_state.demo_resp_idx
    st.session_state.demo_resp_idx = (idx + 1) % len(DEMO_RESPONSES)
    snippet = user_text.strip()
    snippet = snippet if len(snippet) <= 80 else snippet[:77] + "..."
    base = DEMO_RESPONSES[idx]
    notice = ""
    if st.session_state.demo_mode_forced and not st.session_state.demo_notice_shown:
        notice = "Quick heads-up: my live AI assistant is paused, so you're in Demo mode for now. "
        st.session_state.demo_notice_shown = True
    return f'{notice}{base} (Got it: "{snippet}").'


def process_preferences(user_text: str, use_demo: bool, model: str, max_tokens: int) -> bool:
    if use_demo:
        update_prefs(heuristic_prefs(user_text))
        return True
    try:
        extracted = llm_extract_preferences(user_text, model, max_tokens)
        update_prefs(extracted)
        return True
    except Exception as exc:
        enable_forced_demo("Preference extraction failed.")
        update_prefs(heuristic_prefs(user_text))
        return False


def generate_assistant_reply(user_text: str, use_demo: bool, model: str, max_tokens: int) -> str:
    if use_demo:
        return scripted_demo_response(user_text)
    try:
        return llm_chat_response(model, max_tokens)
    except Exception as exc:
        enable_forced_demo("Chat response failed.")
        return scripted_demo_response(user_text)


def filled_pref_count() -> int:
    return sum(1 for v in st.session_state.prefs.values() if v)


def build_pref_summary():
    parts = []
    for key, label in [
        ("riding_style", "style"),
        ("locking_mechanism", "locking"),
        ("thickness", "thickness"),
        ("damping_level", "damping"),
        ("durability", "durability"),
    ]:
        val = st.session_state.prefs.get(key, "")
        if val:
            parts.append(f"{label}: {val}")
    return ", ".join(parts)


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="ODI MTB Chatbot", layout="wide")
st.title("ODI MTB Grip Recommender Chatbot")

try:
    df, csv_path = load_default_dataframe()
except Exception as e:
    st.error(str(e))
    st.stop()

init_session_state()

with st.sidebar:
    st.caption(f"Dataset loaded from: `{csv_path}`")
    if st.button("Restart chat", use_container_width=True):
        reset_chat()
        st.rerun()

    st.session_state.low_cost_mode = st.toggle(
        "Low-cost mode",
        value=st.session_state.get("low_cost_mode", False),
        help="Use a budget-friendly model and shorter replies.",
    )

    manual_demo = st.toggle(
        "Demo mode (no API calls)",
        value=st.session_state.get("demo_mode_manual", False),
        help="Skip OpenRouter and use scripted guidance.",
    )
    st.session_state.demo_mode_manual = manual_demo

    if st.session_state.demo_mode_forced:
        st.info("Demo mode locked in due to an earlier API hiccup.")

    st.markdown("### Current preferences")
    st.json(st.session_state.prefs)

use_demo_mode = st.session_state.demo_mode_manual or st.session_state.demo_mode_forced

model = BUDGET_MODEL if st.session_state.low_cost_mode else PREMIUM_MODEL
chat_max_tokens = 160 if st.session_state.low_cost_mode else 320
pref_max_tokens = 120 if st.session_state.low_cost_mode else 200

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_text = st.chat_input("Tell me what you're feeling on the bike...")

if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})

    extraction_demo = use_demo_mode
    pref_success = process_preferences(user_text, extraction_demo, model, pref_max_tokens)
    if not pref_success:
        use_demo_mode = True

    assistant_reply = generate_assistant_reply(user_text, use_demo_mode, model, chat_max_tokens)
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    st.rerun()

if filled_pref_count() >= 2:
    st.divider()
    st.subheader("Grip ideas based on what you've shared")
    summary = build_pref_summary()
    if summary:
        st.caption(f"Here's what I'm prioritizing: {summary}.")

    recs = recommend(df, st.session_state.prefs, top_n=3)
    for idx, rec in enumerate(recs.to_dict("records"), start=1):
        name = rec.get("name", "Unknown")
        with st.container():
            st.markdown(f"**{idx}. {name}**")
            st.write(
                f"- Style: {rec.get('riding_style', 'N/A')}\n"
                f"- Locking: {rec.get('locking_mechanism', 'N/A')}\n"
                f"- Thickness: {rec.get('thickness', 'N/A')}\n"
                f"- Damping: {rec.get('damping_level', 'N/A')}\n"
                f"- Durability: {rec.get('durability', 'N/A')}\n"
                f"- Pattern: {rec.get('grip_pattern', 'N/A')}\n"
                f"- Ergonomics: {rec.get('ergonomics', 'N/A')}\n"
                f"- Price: {rec.get('price', 'N/A')}\n"
                f"- Key features: {rec.get('key_features', 'N/A')}\n"
                f"- Colors: {rec.get('colors', 'N/A')}"
            )
            st.link_button("Search on ODI", odi_search_link(name), use_container_width=False)

    st.success(
        "Ready to tweak the answers? Adjust what you tell me or hit **Restart chat** in the sidebar for a fresh conversation."
    )
