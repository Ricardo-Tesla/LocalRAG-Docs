import streamlit as st
import requests

# Frontend and backend are intentionally separate processes. This file only
# talks to the API over HTTP, the same way any external client would —
# it has no direct access to the RAG pipeline's Python functions.
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="LocalRAG Docs",
    page_icon="📄",
    layout="wide",
)

# Upload lives in the sidebar rather than the main flow: ingesting a document
# is an occasional action, while asking questions is the primary, repeated
# one and deserves the full viewport.
with st.sidebar:
    st.header("Documents")
    st.caption("Upload technical documents to query them.")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded_file is not None:
        if st.button("Ingest document", use_container_width=True):
            with st.spinner("Processing..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                response = requests.post(f"{API_URL}/upload", files=files)

            if response.status_code == 200:
                st.success(f"Added: {response.json()['filename']}")
            else:
                st.error(f"Upload failed: {response.text}")

st.title("LocalRAG Docs")
st.caption("Ask questions about your uploaded documents. Answers are grounded in your files, with sources shown below.")

question = st.text_input(
    "Ask a question",
    placeholder="e.g. What are the key responsibilities described in this document?",
    label_visibility="collapsed",
)

ask_clicked = st.button("Ask", type="primary")

if ask_clicked:
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Thinking..."):
            response = requests.post(f"{API_URL}/query", json={"question": question})

        if response.status_code == 200:
            result = response.json()

            st.markdown("### Answer")
            st.write(result["answer"])

            st.markdown("### Sources")
            for source in result["sources"]:
                # Expander keeps the page scannable — full chunk text is
                # available on demand rather than dumped inline by default.
                with st.expander(f"{source['source_file']} — Page {source['page']} · relevance {source['similarity_score']}"):
                    st.write(source["text"])
        else:
            st.error(f"Query failed: {response.text}")