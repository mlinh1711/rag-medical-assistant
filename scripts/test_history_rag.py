import sys
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from backend.core.history_rag import run_rag_with_history


def pretty_print_history(history):
    print("---- Conversation history ----")
    for i, msg in enumerate(history, start=1):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        print(f"{i}. {role}: {content}")
    print("------------------------------\n")


def pretty_print_sources(sources):
    if not sources:
        print("No sources returned.\n")
        return

    print("---- Sources ----")
    for s in sources:
        src = s.get("source")
        page = s.get("page")
        snippet = (s.get("snippet") or "")[:250].replace("\n", " ")
        print(f"- {src} (page {page})")
        print(f"  Snippet: {snippet}...")
        print()
    print("-----------------\n")


def main():
    # Fake conversation history for testing history-aware RAG
    history = [
        {
            "role": "user",
            "content": "My child has a fever and the doctor recommended paracetamol.",
        },
        {
            "role": "assistant",
            "content": "Paracetamol is commonly used to treat fever and mild to moderate pain.",
        },
        {
            "role": "user",
            "content": "We already talked about the adult dosage.",
        },
    ]

    question = "What about the dosage?"

    print("====================================================")
    print(" History-aware RAG – Test Script")
    print(" Timestamp:", datetime.now().isoformat())
    print("====================================================\n")

    pretty_print_history(history)

    print(f"Original question: {question}\n")

    # Call history-aware RAG pipeline
    answer, sources, rewritten = run_rag_with_history(
        history=history,
        question=question,
        top_k=5,
        score_threshold=0.27,
    )

    print("Rewritten question:")
    print(rewritten)
    print("\nAnswer:")
    print(answer)
    print()

    pretty_print_sources(sources)

    print("=== End of history RAG test ===")


if __name__ == "__main__":
    main()
