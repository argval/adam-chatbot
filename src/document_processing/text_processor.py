import logging
from pathlib import Path
from typing import Dict, List
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from .document_processor import DocumentProcessor


class TextProcessor(DocumentProcessor):
    def process(
        self, file_path: str, chunk_size: int, chunk_overlap: int
    ) -> Dict[str, List[Document]]:
        try:
            loader = UnstructuredFileLoader(file_path, mode="single")
            docs = loader.load()

            if not docs:
                logging.warning(
                    f"UnstructuredFileLoader returned no documents for {file_path}."
                )
                return {"chunks": [], "classes": []}

            # Hybrid approach: Use header splitting for Markdown/HTML, character for others
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            character_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )

            # Check file extension for hybrid logic
            ext = file_path.split(".")[-1].lower()
            if ext in ["md", "html"]:
                chunks = (
                    markdown_splitter.split_text(docs[0].page_content) if docs else []
                )
            else:
                chunks = character_splitter.split_documents(docs)

            for chunk in chunks:
                # Ensure downstream consumers always receive a source pointer.
                chunk.metadata["source"] = file_path
                chunk.metadata["source_name"] = Path(file_path).name

            return {"chunks": chunks, "classes": []}
        except Exception as e:
            logging.error(f"Failed to process text file {file_path}. Error: {e}")
            return {"chunks": [], "classes": []}
