import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def create_or_get_vector_store(chunks=None, persist_directory="./db"):
    """
    Creates or loads a ChromaDB vector store using local CPU HuggingFace embeddings.
    """
    # Runs lightweight embedding generation locally on your CPU
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if chunks:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        return vector_store
    else:
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        return vector_store