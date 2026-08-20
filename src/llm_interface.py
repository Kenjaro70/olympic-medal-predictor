"""LLM-powered natural language interface for the Olympic medal model.

Flow: user question -> LLM extracts structured features (JSON) -> validation
(missing / out-of-scope handling) -> trained model inference -> LLM writes a
conversational answer around the prediction.

The LLM call is injectable (``llm`` argument) so the parsing and edge-case
logic is unit-testable without network access. The real client talks to any
OpenAI-compatible endpoint; Nebius AI Studio is the default provider.
API keys come from environment variables only (see .env.example).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Callable

import joblib

from src.preprocess import transform_inference

DEFAULT_BASE_URL = "https://api.studio.nebius.com/v1/"
DEFAULT_MODEL = "meta-llama/Meta-Llama-3.3-70B-Instruct"
MODEL_BUNDLE_PATH = Path("models/model_bundle.joblib")

REQUIRED_FEATURES = ["sex", "age", "noc", "sport"]

WINTER_SPORTS = {
    "Alpine Skiing", "Biathlon", "Bobsleigh", "Cross Country Skiing",
    "Curling", "Figure Skating", "Freestyle Skiing", "Ice Hockey", "Luge",
    "Nordic Combined", "Short Track Speed Skating", "Skeleton",
    "Ski Jumping", "Snowboarding", "Speed Skating",
}

EXTRACTION_SYSTEM_PROMPT = """\
You extract structured features from a user's question about Olympic medal
chances. Respond with ONLY a JSON object, no prose, with exactly these keys:

- "sex": "M" or "F", or null if not stated or inferable
- "age": integer age in years, or null
- "height": height in centimeters (convert from feet/inches if needed), or null
- "weight": weight in kilograms (convert from pounds if needed), or null
- "noc": the 3-letter IOC country code for the athlete's country
  (e.g. United States -> "USA", Great Britain -> "GBR", Japan -> "JPN"),
  or null
- "sport": the canonical Olympic sport name in English, capitalized
  (e.g. swimmer -> "Swimming", track and field/runner -> "Athletics",
  gymnast -> "Gymnastics", weightlifter -> "Weightlifting"), or null
- "season": "Summer" or "Winter" based on the sport, or null if unknown
- "year": the Olympic year mentioned, or null
- "out_of_scope": true ONLY if the question is not about predicting an
  athlete's Olympic medal chances (e.g. general trivia, medical advice,
  unrelated chit-chat); otherwise false

Do not guess values that are not stated or clearly implied. Use null.
"""

RESPONSE_SYSTEM_PROMPT = """\
You are the voice of an Olympic medal prediction tool. You are given the
user's question, the parsed athlete features, and the trained model's
predicted medal probability. Write a short, friendly, conversational answer
that:
1. States the predicted probability of winning a medal as a percentage.
2. Explains the result in context using the features (country's historical
   medal rate and sport are the strongest signals in this model).
3. Mentions which provided details mattered and notes any values that were
   assumed/imputed because the user did not provide them.
4. Ends with a one-sentence caveat: the model is trained on historical
   Olympic data (1896-2016) and predicts base rates, not individual talent,
   so it cannot account for personal skill or qualification.
Keep it under 180 words. Do not invent numbers beyond those given.
"""


def get_client():
    """Build an OpenAI-compatible client for Nebius AI Studio (or override)."""
    from openai import OpenAI

    api_key = os.environ.get("NEBIUS_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No API key found. Set NEBIUS_API_KEY (or OPENAI_API_KEY) in your "
            "environment or .env file — see .env.example."
        )
    base_url = os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)
    return OpenAI(api_key=api_key, base_url=base_url)


def call_llm(messages: list[dict], temperature: float = 0.2) -> str:
    """Default LLM callable: chat-completion against the configured provider."""
    client = get_client()
    model = os.environ.get("LLM_MODEL", DEFAULT_MODEL)
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature
    )
    return response.choices[0].message.content


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of an LLM response, tolerating fences."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM output: {text[:200]!r}")
    return json.loads(match.group(0))


def normalize_features(raw: dict) -> dict:
    """Coerce and sanity-check the LLM-extracted features."""
    features: dict = {
        "sex": None, "age": None, "height": None, "weight": None,
        "noc": None, "sport": None, "season": None, "year": None,
        "out_of_scope": bool(raw.get("out_of_scope", False)),
    }
    sex = raw.get("sex")
    if isinstance(sex, str) and sex.strip().upper() in ("M", "F"):
        features["sex"] = sex.strip().upper()
    for key in ("age", "height", "weight", "year"):
        value = raw.get(key)
        if isinstance(value, (int, float)) and value > 0:
            features[key] = float(value)
    noc = raw.get("noc")
    if isinstance(noc, str) and len(noc.strip()) == 3:
        features["noc"] = noc.strip().upper()
    sport = raw.get("sport")
    if isinstance(sport, str) and sport.strip():
        features["sport"] = sport.strip().title()
    season = raw.get("season")
    if isinstance(season, str) and season.strip().title() in ("Summer", "Winter"):
        features["season"] = season.strip().title()
    elif features["sport"]:
        features["season"] = (
            "Winter" if features["sport"] in WINTER_SPORTS else "Summer"
        )
    if features["year"] is not None:
        # Clamp to the range the model was trained on.
        features["year"] = float(min(max(features["year"], 1896), 2016))
    if features["age"] is not None and not (10 <= features["age"] <= 97):
        features["age"] = None
    return features


def parse_query(text: str, llm: Callable[[list[dict]], str] | None = None) -> dict:
    """Extract normalized model features from a natural language question."""
    llm = llm or call_llm
    output = llm(
        [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
    )
    return normalize_features(extract_json(output))


def missing_required(features: dict) -> list[str]:
    return [f for f in REQUIRED_FEATURES if features.get(f) is None]


def load_bundle(path: Path | str = MODEL_BUNDLE_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — train the model first: python -m src.train"
        )
    return joblib.load(path)


def predict_from_features(features: dict, bundle: dict) -> dict:
    """Run the trained model on one parsed feature dict."""
    row = transform_inference(features, bundle["preprocessing"])
    proba = float(bundle["model"].predict_proba(row)[0, 1])
    known_nocs = bundle["preprocessing"]["encoders"]["noc_rates"]
    known_sports = bundle["preprocessing"]["encoders"]["sport_rates"]
    return {
        "medal_probability": proba,
        "prediction": int(proba >= 0.5),
        "noc_known": features.get("noc") in known_nocs,
        "sport_known": features.get("sport") in known_sports,
        "imputed": [
            k for k in ("height", "weight", "year") if features.get(k) is None
        ],
    }


FIELD_QUESTIONS = {
    "sex": "the athlete's sex (M/F)",
    "age": "the athlete's age",
    "noc": "the athlete's country",
    "sport": "the sport they compete in",
}


def clarification_message(missing: list[str]) -> str:
    needs = ", ".join(FIELD_QUESTIONS[m] for m in missing)
    return (
        "I'd love to estimate that, but I'm missing a few details the model "
        f"needs: {needs}. Could you tell me? Height, weight, and the Olympic "
        "year are optional but improve the estimate."
    )


OUT_OF_SCOPE_MESSAGE = (
    "I'm a focused tool: I predict the probability that an athlete wins an "
    "Olympic medal, based on sex, age, country, sport, and body metrics. "
    "That question is outside what my model can answer — try something like: "
    "\"What are the medal chances for a 24-year-old American female swimmer?\""
)


def answer(
    query: str,
    bundle: dict | None = None,
    llm: Callable[[list[dict]], str] | None = None,
) -> dict:
    """End-to-end pipeline for one user question.

    Returns a dict with 'reply' plus the intermediate 'features' and
    'prediction' (None when the query never reached the model).
    """
    llm = llm or call_llm
    try:
        features = parse_query(query, llm)
    except (ValueError, json.JSONDecodeError):
        return {
            "reply": (
                "Sorry — I couldn't understand that. Try describing the "
                "athlete, e.g. \"25-year-old male sprinter from Jamaica, "
                "180 cm, 77 kg — what are his medal odds?\""
            ),
            "features": None,
            "prediction": None,
        }

    if features["out_of_scope"]:
        return {"reply": OUT_OF_SCOPE_MESSAGE, "features": features, "prediction": None}

    missing = missing_required(features)
    if missing:
        return {
            "reply": clarification_message(missing),
            "features": features,
            "prediction": None,
        }

    if bundle is None:
        bundle = load_bundle()
    prediction = predict_from_features(features, bundle)

    context = {
        "parsed_features": {k: v for k, v in features.items() if k != "out_of_scope"},
        "medal_probability": round(prediction["medal_probability"], 4),
        "assumed_or_imputed_fields": prediction["imputed"],
        "country_seen_in_training": prediction["noc_known"],
        "sport_seen_in_training": prediction["sport_known"],
        "model_info": bundle["metadata"],
    }
    reply = llm(
        [
            {"role": "system", "content": RESPONSE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"User question: {query}\n\n"
                    f"Model output and context:\n{json.dumps(context, indent=2)}"
                ),
            },
        ]
    )
    return {"reply": reply, "features": features, "prediction": prediction}
