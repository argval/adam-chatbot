import logging
from pathlib import Path
from typing import Dict, List, Optional

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document

from .document_processor import DocumentProcessor


class TextProcessor(DocumentProcessor):
    def process(
        self, file_path: str, chunk_size: int, chunk_overlap: int
    ) -> Dict[str, List[Document]]:
        try:
            docs = self._load_documents(file_path)

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

    def _load_documents(self, file_path: str) -> List[Document]:
        """
        Attempt to load documents using UnstructuredFileLoader. If unavailable or failing,
        fall back to a plain text read.
        """
        try:
            loader = UnstructuredFileLoader(file_path, mode="single")
            return loader.load()
        except Exception as exc:
            logging.warning(
                "Falling back to plain-text loader for %s due to %s", file_path, exc
            )
            content = self._read_text_file(file_path)
            if content is None:
                return []
            return [Document(page_content=content, metadata={"source": file_path})]

    @staticmethod
    def _read_text_file(file_path: str) -> Optional[str]:
        path = Path(file_path)
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                return path.read_text(encoding="latin-1")
            except Exception as exc:
                logging.error("Unable to read %s due to %s", file_path, exc)
                return None
        except Exception as exc:
            logging.error("Unable to read %s due to %s", file_path, exc)
            return None
