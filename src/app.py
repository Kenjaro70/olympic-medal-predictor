"""Streamlit chat interface for the Olympic medal predictor.

Run with:
    streamlit run src/app.py
"""

import os
import sys
import urllib.request
from pathlib import Path

# Allow `streamlit run src/app.py` from the repo root to find the src package.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)  # model/config paths in the project are repo-relative

import streamlit as st
from dotenv import load_dotenv

from src.llm_interface import MODEL_BUNDLE_PATH, answer, load_bundle

load_dotenv()

DATA_URL = (
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/"
    "master/data/2021/2021-07-27/olympics.csv"
)
# Trained on first boot when no bundle is present (e.g. on Streamlit
# Community Cloud, which starts from a bare git clone). hist_gb_tuned is
# fast to train and produces a compact model that fits cloud memory limits.
BOOTSTRAP_EXPERIMENT = "hist_gb_tuned"


def ensure_model() -> None:
    """Download the public dataset and train a model if no bundle exists."""
    if MODEL_BUNDLE_PATH.exists():
        return
    from src.train import load_config, run_experiments

    config = load_config("configs/config.yaml")
    raw_path = Path(config["data"]["raw_path"])
    if not raw_path.exists():
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with st.spinner("First run: downloading the Olympic dataset (~36 MB)..."):
            urllib.request.urlretrieve(DATA_URL, raw_path)
    # Training also tunes the decision threshold, which fits one probe model
    # per walk-forward fold before the final fit -- four fits, not one.
    with st.spinner("First run: training the medal model (a minute or two)..."):
        run_experiments(config, only=BOOTSTRAP_EXPERIMENT)

st.set_page_config(page_title="Olympic Medal Predictor", page_icon="🥇")
st.title("🥇 Olympic Medal Predictor")
st.caption(
    "Describe an athlete in plain English and I'll estimate their medal "
    "chances with a model trained on 120 years of Olympic history."
)


@st.cache_resource
def get_bundle():
    ensure_model()
    return load_bundle()


try:
    bundle = get_bundle()
except Exception as exc:  # missing bundle, failed download, or training error
    st.error(f"Could not load or train the model: {exc}")
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
