"""Minimal command-line chat loop (alternative to the Streamlit app).

Usage:
    python -m src.cli                     # interactive loop
    python -m src.cli "your question"     # one-shot
"""

import sys

from dotenv import load_dotenv

from src.llm_interface import answer, load_bundle


def main() -> None:
    load_dotenv()
    bundle = load_bundle()
    if len(sys.argv) > 1:
        result = answer(" ".join(sys.argv[1:]), bundle)
        print(result["reply"])
        return
    print("Olympic Medal Predictor — type a question, or 'quit' to exit.\n")
    while True:
        try:
            query = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in ("quit", "exit"):
            break
        result = answer(query, bundle)
        print(f"\nbot> {result['reply']}\n")


if __name__ == "__main__":
    main()
