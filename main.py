"""Local RAG Q&A assistant (Plan Phase 2, Week 4).

Run `python ingest.py` first to populate the knowledge base, then run
this file to ask questions. Everything runs on-device via Foundry Local:
no internet connection is needed once the models are downloaded.

Pass --debug to see, for every question, which chunks were retrieved
(source, a text preview, and their similarity score) and whether each
one cleared MIN_SCORE - useful for tuning MIN_SCORE or the chunker.
"""

import sys

from foundry_local_sdk import Configuration, FoundryLocalManager

import db
from retrieve import get_scored_candidates, get_top_chunks

EMBEDDING_MODEL_ALIAS = "qwen3-embedding-0.6b"
CHAT_MODEL_ALIAS = "phi-4-mini"
TOP_K = 3
MIN_SCORE = 0.45
MAX_ANSWER_TOKENS = 500
DEBUG = "--debug" in sys.argv

SYSTEM_PROMPT_TEMPLATE = (
    "Answer the user's question using ONLY the provided context. "
    "You may combine or draw conclusions from facts that are stated in the "
    "context - that kind of inference is fine. But you may not introduce "
    "any fact, definition, explanation, or example that is not present in "
    "the context, even if you know it from elsewhere and even if it seems "
    "like helpful background. "
    "If the context only partially answers the question, answer just the "
    "part it supports and explicitly say what is missing - do not fill the "
    "gap with outside knowledge. If the context doesn't contain enough "
    "information to answer at all, say you don't know rather than guessing. "
    "When you do answer, mention which source document the information came from.\n\n"
    "Context:\n{context}"
)


def build_context(chunks: list[dict]) -> str:
    if not chunks:
        return "(no relevant context found)"
    return "\n\n".join(f"[{c['source']}] {c['content']}" for c in chunks)


def print_debug_candidates(question: str, embedding_client) -> None:
    """Show the top candidate chunks and their scores, pass/fail against MIN_SCORE."""
    candidates = get_scored_candidates(embedding_client, question, top_k=5)
    print(f"\n[debug] Query: {question!r}  (MIN_SCORE={MIN_SCORE})")
    if not candidates:
        print("[debug] No chunks in the knowledge base at all.")
        return
    for c in candidates:
        status = "PASS" if c["score"] >= MIN_SCORE else "fail"
        preview = c["content"][:80].replace("\n", " ")
        print(f"[debug] {status}  score={c['score']:.3f}  source={c['source']}  \"{preview}...\"")
    print()


def print_debug_full_context(context: str) -> None:
    """Print the exact context text handed to the chat model, in full.

    This is the only reliable way to check groundedness: compare each
    claim in the printed Answer against this text. If a claim in the
    answer isn't in here, it didn't come from the book - regardless of
    what the model's system prompt asked for.
    """
    print("[debug] ---- Full context sent to the model ----")
    print(context)
    print("[debug] ---- End of context ----\n")


def answer_query(chat_client, embedding_client, question: str) -> None:
    """Retrieve relevant chunks and stream a grounded answer to stdout."""
    if DEBUG:
        print_debug_candidates(question, embedding_client)

    top_chunks = get_top_chunks(embedding_client, question, top_k=TOP_K, min_score=MIN_SCORE)

    if not top_chunks:
        print("Answer: I don't have information about that in the knowledge base.\n")
        return

    context = build_context(top_chunks)
    if DEBUG:
        print_debug_full_context(context)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(context=context)},
        {"role": "user", "content": question},
    ]

    print("Answer: ", end="", flush=True)
    for chunk in chat_client.complete_streaming_chat(messages):
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
    print("\n")


def main() -> None:
    conn = db.get_connection()
    db.init_db(conn)
    n_chunks = db.count_chunks(conn)
    conn.close()

    if n_chunks == 0:
        print("Knowledge base is empty. Run `python ingest.py` first, then try again.")
        return

    print(f"Knowledge base has {n_chunks} chunk(s). Loading models...")
    if DEBUG:
        print("Debug mode ON - retrieval scores will be shown for each question.")

    config = Configuration(app_name="foundry_local_rag")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    embedding_model = manager.catalog.get_model(EMBEDDING_MODEL_ALIAS)
    embedding_model.download(
        lambda p: print(f"\rDownloading embedding model: {p:.1f}%", end="", flush=True)
    )
    print()
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    chat_model = manager.catalog.get_model(CHAT_MODEL_ALIAS)
    chat_model.download(
        lambda p: print(f"\rDownloading chat model: {p:.1f}%", end="", flush=True)
    )
    print()
    chat_model.load()
    chat_client = chat_model.get_chat_client()
    chat_client.settings.max_tokens = MAX_ANSWER_TOKENS

    print("\nModels loaded. Ready for questions.")
    print("Example questions:")
    print('  "What programming languages does the Foundry Local SDK support?"')
    print('  "What is retrieval-augmented generation?"')
    print('\nType "quit" to exit.\n')

    try:
        while True:
            question = input("Question: ").strip()
            if not question or question.lower() == "quit":
                break
            answer_query(chat_client, embedding_client, question)
    finally:
        embedding_model.unload()
        chat_model.unload()
        print("Models unloaded. Done!")


if __name__ == "__main__":
    main()
