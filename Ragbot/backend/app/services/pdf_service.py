from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def extract_text(pdf_path):

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    chunk_size = min(1500, max(500, len(pages) * 50))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
    chunks = []

    for i, page in enumerate(pages):
        splits = text_splitter.split_text(page.page_content)
        for split in splits:
            chunks.append(
                Document(
                    page_content=split,
                    metadata={"page_number": i + 1}
                )
            )

    print(f"No of Chunks: {len(chunks)}")
    return chunks

