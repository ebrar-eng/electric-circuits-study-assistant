"""Streamlit web UI for the local RAG assistant.

Reuses the same retrieval (db.py, retrieve.py) and grounding logic as
main.py's CLI - this is just a different front end. Run with:

    streamlit run app.py

Run `python ingest.py` first if you haven't already, to populate the
knowledge base.
"""

import streamlit as st
from foundry_local_sdk import Configuration, FoundryLocalManager

import db
from retrieve import get_scored_candidates, get_top_chunks

EMBEDDING_MODEL_ALIAS = "qwen3-embedding-0.6b"
CHAT_MODEL_ALIAS = "phi-4-mini"
TOP_K = 3
MIN_SCORE = 0.45
MAX_ANSWER_TOKENS = 500

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


@st.cache_resource(show_spinner=False)
def load_models():
    """Load both models once per server process and keep them resident.

    st.cache_resource means this runs once no matter how many times the
    script re-executes on user interaction (Streamlit re-runs the whole
    script on every input) - without it, every question would reload
    both models from scratch.
    """
    config = Configuration(app_name="foundry_local_rag")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    embedding_model = manager.catalog.get_model(EMBEDDING_MODEL_ALIAS)
    embedding_model.download(lambda p: None)
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    chat_model = manager.catalog.get_model(CHAT_MODEL_ALIAS)
    chat_model.download(lambda p: None)
    chat_model.load()
    chat_client = chat_model.get_chat_client()
    chat_client.settings.max_tokens = MAX_ANSWER_TOKENS

    return embedding_client, chat_client


def build_context(chunks: list[dict]) -> str:
    if not chunks:
        return "(no relevant context found)"
    return "\n\n".join(f"[{c['source']}] {c['content']}" for c in chunks)


def get_answer(chat_client, embedding_client, question: str):
    """Returns (answer_text, top_chunks, context) for display."""
    top_chunks = get_top_chunks(embedding_client, question, top_k=TOP_K, min_score=MIN_SCORE)

    if not top_chunks:
        return "I don't have information about that in the knowledge base.", [], ""

    context = build_context(top_chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(context=context)},
        {"role": "user", "content": question},
    ]

    answer_parts = []
    for chunk in chat_client.complete_streaming_chat(messages):
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if content:
            answer_parts.append(content)

    return "".join(answer_parts), top_chunks, context


def main():
    st.set_page_config(page_title="Electric Circuits Study Assistant", page_icon="⚡")
    st.title("⚡ Electric Circuits Study Assistant")
    st.caption(
        "Answers grounded in *Electric Circuits*, 10th Edition (Nilsson & Riedel) — "
        "runs fully offline via Foundry Local."
    )

    conn = db.get_connection()
    db.init_db(conn)
    n_chunks = db.count_chunks(conn)
    conn.close()

    if n_chunks == 0:
        st.warning("Knowledge base is empty. Run `python ingest.py` first, then reload this page.")
        st.stop()

    st.sidebar.metric("Chunks in knowledge base", n_chunks)
    debug_mode = st.sidebar.toggle("Show retrieval debug info", value=False)
    st.sidebar.caption(f"Chat model: {CHAT_MODEL_ALIAS}\n\nEmbedding model: {EMBEDDING_MODEL_ALIAS}\n\nMIN_SCORE: {MIN_SCORE}")

    with st.spinner("Loading models (first run downloads them - this can take a while)..."):
        embedding_client, chat_client = load_models()

    if "history" not in st.session_state:
        st.session_state.history = []

    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn["role"] == "assistant" and debug_mode and turn.get("debug"):
                with st.expander("Retrieval debug info"):
                    for c in turn["debug"]["candidates"]:
                        status = "✅" if c["score"] >= MIN_SCORE else "❌"
                        st.text(f"{status} score={c['score']:.3f}  {c['source']}")
                        st.caption(c["content"][:200])
                    st.text("Full context sent to model:")
                    st.text(turn["debug"]["context"] or "(none - below threshold)")

    question = st.chat_input("Ask a question about Electric Circuits...")
    if question:
        st.session_state.history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                candidates = get_scored_candidates(embedding_client, question, top_k=5) if debug_mode else []
                answer, top_chunks, context = get_answer(chat_client, embedding_client, question)
            st.markdown(answer)
            debug_info = {"candidates": candidates, "context": context} if debug_mode else None
            if debug_mode:
                with st.expander("Retrieval debug info"):
                    for c in candidates:
                        status = "✅" if c["score"] >= MIN_SCORE else "❌"
                        st.text(f"{status} score={c['score']:.3f}  {c['source']}")
                        st.caption(c["content"][:200])
                    st.text("Full context sent to model:")
                    st.text(context or "(none - below threshold)")

        st.session_state.history.append({"role": "assistant", "content": answer, "debug": debug_info})


if __name__ == "__main__":
    main()
