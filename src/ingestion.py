from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

COLLECTION_NAME = "rag_documents"


def clean_text(text: str) -> str:
    """Clean extracted PDF text while preserving useful line breaks."""
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = "".join(
        char for char in text
        if not (0xD800 <= ord(char) <= 0xDFFF)
    )
    text = "".join(
        char
        for char in text
        if unicodedata.category(char) not in {"Cc", "Cf"}
        or char in "\n\t"
    )
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def load_and_chunk_pdf(pdf_path: str | Path):
    """
    Load a PDF, split by Markdown headings, then split oversized sections.

    MarkdownHeaderTextSplitter only recognizes headings explicitly represented
    as Markdown (for example, '## Pricing' and '### How much does it cost?').
    PyMuPDFLoader does not automatically add Markdown markers based on visual
    font sizes in a PDF.
    """
    pages = PyMuPDFLoader(str(pdf_path)).load()

    for page in pages:
        page.page_content = clean_text(page.page_content)

    # Combine pages so headings and FAQ entries can span page boundaries.
    text = "\n\n".join(
        page.page_content for page in pages if page.page_content
    )

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("##", "section"),
            ("###", "question"),
        ],
        strip_headers=False,
    )
    sections = header_splitter.split_text(text)

    # Split oversized sections into smaller chunks.
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
    )
    chunks = size_splitter.split_documents(sections)

    for index, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = index
        chunk.metadata["source"] = str(pdf_path)

    return chunks


def build_vector_store(
    pdf_path: str | Path,
    embeddings,
    persist_directory: str | Path = "chroma_db",
) -> Chroma:
    """Index a PDF locally; Chroma persists rather than using temporary storage."""
    chunks = load_and_chunk_pdf(pdf_path)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_directory),
    )


def load_vector_store(
    embeddings,
    persist_directory: str | Path = "chroma_db",
) -> Chroma:
    """Open an existing local index created by build_vector_store."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )
