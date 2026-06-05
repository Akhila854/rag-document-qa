import streamlit as st
import requests

import os
API_URL = os.getenv("API_URL", "http://localhost:8080")

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Document Q&A with RAG")
st.caption("Upload a PDF and ask questions about it using AI")

# --- Session state ---
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False
if "filename" not in st.session_state:
    st.session_state.filename = None
if "history" not in st.session_state:
    st.session_state.history = []

# --- Upload section ---
st.subheader("1. Upload a PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file and not st.session_state.pdf_loaded:
    with st.spinner("Processing PDF — extracting, chunking, embedding..."):
        response = requests.post(
            f"{API_URL}/upload",
            files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        )
    if response.status_code == 200:
        data = response.json()
        st.session_state.pdf_loaded = True
        st.session_state.filename = uploaded_file.name
        st.success(f"Ready — {data['pages']} pages, {data['chunks']} chunks indexed")
    else:
        st.error(f"Upload failed: {response.text}")

if st.session_state.pdf_loaded:
    st.caption(f"Loaded: `{st.session_state.filename}`")

# --- Q&A section ---
st.subheader("2. Ask a Question")
question = st.text_input(
    "Your question",
    placeholder="e.g. What is the attention mechanism?",
    disabled=not st.session_state.pdf_loaded
)

if st.button("Ask", disabled=not st.session_state.pdf_loaded or not question):
    with st.spinner("Searching document and generating answer..."):
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": question}
        )
    if response.status_code == 200:
        result = response.json()
        st.session_state.history.append(result)
    else:
        st.error(f"Error: {response.text}")

# --- Display history ---
if st.session_state.history:
    st.subheader("3. Answers")
    for item in reversed(st.session_state.history):
        with st.container():
            st.markdown(f"**Q: {item['question']}**")
            st.markdown(item["answer"])

            with st.expander("View sources"):
                for s in item["sources"]:
                    st.markdown(f"**Page {s['page']}** (score: `{s['score']}`)")
                    st.caption(s["text"][:300] + "...")
            st.divider()