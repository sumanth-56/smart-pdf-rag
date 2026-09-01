import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_pdf(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Loads a PDF file and splits it into overlapping semantic text chunks.
    
    Args:
        file_path (str): Path to the PDF file.
        chunk_size (int): Max number of characters per chunk.
        chunk_overlap (int): Overlap character count to maintain sentence context.
        
    Returns:
        list: List of LangChain Document objects containing page content and metadata.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at: {file_path}")

    # Step 1: Extract text page-by-page from the PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Step 2: Split text into structured overlapping chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks