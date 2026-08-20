"""Streamlit chat interface for the Olympic medal predictor.

Run with:
    streamlit run src/app.py
"""

import os
import sys
from pathlib import Path

# Allow `streamlit run src/app.py` from the repo root to find the src package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from dotenv import load_dotenv

from src.llm_interface import answer, load_bundle

load_dotenv()

st.set_page_config(page_title="Olympic Medal Predictor", page_icon="🥇")
st.title("🥇 Olympic Medal Predictor")
st.caption(
    "Describe an athlete in plain English and I'll estimate their medal "
    "chances with a model trained on 120 years of Olympic history."
)


@st.cache_resource
def get_bundle():
    return load_bundle()


try:
    bundle = get_bundle()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.subheader("Model")
    meta = bundle["metadata"]
    st.write(f"**{meta['experiment_name']}** ({meta['model_type']})")
    st.json(meta["metrics"])
    if not (os.environ.get("NEBIUS_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        st.warning(
            "No LLM API key found. Set NEBIUS_API_KEY in a .env file "
            "(see .env.example)."
        )

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! Ask me something like: *\"I'm a 24-year-old female "
                "swimmer from the USA, 175 cm and 63 kg — what are my medal "
                "chances?\"*"
            ),
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Describe an athlete..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Parsing your question and running the model..."):
            try:
                result = answer(prompt, bundle)
                reply = result["reply"]
                if result["prediction"] is not None:
                    p = result["prediction"]["medal_probability"]
                    st.metric("Medal probability", f"{p:.1%}")
            except Exception as exc:  # surface config/network issues to the user
                reply = f"Something went wrong: {exc}"
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
