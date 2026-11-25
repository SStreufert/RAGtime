# app.py
import streamlit as st

import healthcheck
from rag import answer, _short_source

# doc_title = "Ask your documents"
doc_title = "Swan Lagoon Tourist Assistant"

st.set_page_config(page_title=doc_title, layout="centered")
st.title(doc_title)

# Successes are cached per process; failures re-probe on the next interaction,
# so starting the model server later recovers without restarting Streamlit.
_endpoints = healthcheck.probe_launch(("chat", "embeddings"))
if not healthcheck.all_ok(_endpoints):
    st.error(healthcheck.report_text(_endpoints))
    st.stop()

st.caption("G'day!")
st.caption("Thinking of visiting Beautiful Swan Lagoon? Make your visit even more enjoyable by asking the Swan Lagoon Assistant your travel-related questions.")

with st.form("ask_form", clear_on_submit=False):
    question = st.text_input("Question", placeholder="What destination is a can't miss?")
    k = st.slider("Passages to retrieve", 1, 8, 3)
    submitted = st.form_submit_button("Ask", type="primary")

if submitted and question:
    with st.spinner("Searching and generating…"):
        text, sources, used = answer(question, k=k)

    st.markdown(text)
    st.write("---")
    
    # Trailing block: exactly which retrieved documents the answer relied on.
    # Rendered deterministically from the LLM's declared USED_SOURCES — never
    # placed by the model, so location and format are consistent run to run.
    if used:
        st.markdown("**Page citations:**")
        for i in used:
            if 1 <= i <= len(sources):
                doc = sources[i - 1]
                page = doc.metadata.get("page")
                if isinstance(page, int):
                    label = page + 1
                elif isinstance(page, str) and page.isdigit():
                    label = int(page) + 1
                else:
                    label = "?"
                st.caption(f"[{i}] {_short_source(doc.metadata.get('source'))} — page {label}")
            else:
                st.caption(f"[{i}] Source index out of range")
    else:
        st.caption("Page citations: not cited in answer.")

    # Expander still shows every retrieved passage, used or not.
    with st.expander(f"Retrieved passages ({len(sources)})"):
        for i, doc in enumerate(sources, start=1):
            page = doc.metadata.get("page")
            if isinstance(page, int):
                label = page + 1
            elif isinstance(page, str) and page.isdigit():
                label = int(page) + 1
            else:
                label = "?"
            st.caption(f"[{i}] {_short_source(doc.metadata.get('source'))} — page {label}")
            st.text(doc.page_content[:500])
    st.caption("Answers come only from the PDFs indexed in chroma_db.")
