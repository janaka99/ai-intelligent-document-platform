import json
import io
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Dict, Any, List

try:
    import pypdf
except ImportError:
    pypdf = None

@tool
def extract_key_entities(text: str) -> str:
    """Extract key entities such as names, dates, organizations, and amounts from the document text.
    Returns a JSON string of the extracted entities.
    """
    # In a real scenario, this might call another LLM, a specific NLP model, or use Regex
    # For now, we simulate extraction. The LLM will still use this tool.
    return json.dumps({
        "status": "success",
        "message": "Entities extracted (simulated)",
        "entities": ["Simulated Entity 1", "Simulated Entity 2"]
    })

@tool
def summarize_document(text: str) -> str:
    """Generate a brief summary of the provided document text."""
    # Simulated summarization
    return "This is a simulated summary of the document."

def extract_text_from_bytes(file_bytes: bytes, filename: str) -> str:
    """Extract text from raw file bytes based on file extension."""
    filename = filename.lower()
    
    if filename.endswith(".pdf"):
        if pypdf is None:
            raise ValueError("pypdf is not installed. Cannot parse PDF files.")
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    elif filename.endswith((".txt", ".md", ".csv", ".json")):
        return file_bytes.decode("utf-8", errors="ignore")
        
    else:
        raise ValueError(f"Unsupported file type: {filename}. Please upload a .txt, .md, .csv, or .pdf file.")

def chunk_document_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Helper function to chunk document text using LangChain's RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_text(text)

@tool
def chunk_document_tool(text: str) -> str:
    """Chunk the provided document text into smaller, manageable pieces. 
    Useful when the text is too large to process all at once.
    Returns a JSON string containing the number of chunks and the first chunk as a preview.
    """
    chunks = chunk_document_text(text)
    return json.dumps({
        "status": "success",
        "total_chunks": len(chunks),
        "preview_first_chunk": chunks[0] if chunks else ""
    })

DOCUMENT_TOOLS = [extract_key_entities, summarize_document, chunk_document_tool]
