import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from src.pdf_loader import load_and_chunk_pdf
from src.vector_store import create_or_get_vector_store
from src.rag_chain import get_rag_chain

# Load environment variables (.env file)
load_dotenv()

st.set_page_config(page_title="Smart PDF Analyzer", page_icon="📄", layout="wide")

st.title("📄 Document-Based RAG Assistant")
st.subheader("Analyze and chat with your PDF documents accurately")

# Sidebar for configuration and PDF uploading
with st.sidebar:
    st.header("Document Setup")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    process_btn = st.button("Process & Index PDF")

# Initialize Streamlit session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# Process uploaded PDF when user clicks the button
if process_btn and uploaded_file is not None:
    with st.spinner("Processing PDF (Extracting text & indexing vectors)..."):
        # Save uploaded file temporarily to disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        try:
            # Step 1: Load and chunk text
            chunks = load_and_chunk_pdf(tmp_path)
            
            # Step 2: Store in ChromaDB
            vector_store = create_or_get_vector_store(chunks=chunks)
            
            # Step 3: Initialize RAG Chain
            chain, _ = get_rag_chain(vector_store)
            st.session_state.rag_chain = chain
            
            st.success(f"Indexed {len(chunks)} chunks successfully! Ready to answer questions.")
        except Exception as e:
            st.error(f"Error processing document: {e}")
        finally:
            # Clean up temporary file
            os.remove(tmp_path)

# Display active chat messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User query input
if user_question := st.chat_input("Ask a question about your PDF..."):
    # Render user prompt
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Generate answer if RAG chain is ready
    if st.session_state.rag_chain:
        with st.chat_message("assistant"):
            with st.spinner("Searching document & generating answer..."):
                response = st.session_state.rag_chain.invoke(user_question)
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
    else:
        st.warning("Please upload and process a PDF document in the sidebar first.")