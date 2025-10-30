import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document

from .document_processor import DocumentProcessor


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


class _ImportCache:
    bs4_warning_emitted = False


def _get_bs4() -> Optional[object]:
    try:
        import bs4  # type: ignore

        return bs4
    except ImportError:
        if not _ImportCache.bs4_warning_emitted:
            logging.warning(
                "BeautifulSoup (bs4) is not installed; HTML table/image extraction disabled."
            )
            _ImportCache.bs4_warning_emitted = True
        return None


@dataclass
class TextExtractionConfig:
    enable_html_tables: bool = _env_flag("TEXT_ENABLE_HTML_TABLES", True)
    enable_html_images: bool = _env_flag("TEXT_ENABLE_HTML_IMAGES", True)


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
            if ext in ["md", "html", "htm"]:
                chunks = (
                    markdown_splitter.split_text(docs[0].page_content) if docs else []
                )
            else:
                chunks = character_splitter.split_documents(docs)

            config = TextExtractionConfig()
            extras: List[Document] = []
            if ext in ["html", "htm"]:
                html_content = docs[0].page_content if docs else ""
                if config.enable_html_tables:
                    extras.extend(self._extract_html_tables(html_content, file_path))
                if config.enable_html_images:
                    extras.extend(self._extract_html_images(html_content, file_path))

            for chunk in chunks:
                # Ensure downstream consumers always receive a source pointer.
                chunk.metadata["source"] = file_path
                chunk.metadata["source_name"] = Path(file_path).name

            for extra in extras:
                extra.metadata.setdefault("source", file_path)
                extra.metadata.setdefault("source_name", Path(file_path).name)

            chunks.extend(extras)

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

    def _extract_html_tables(self, html: str, file_path: str) -> List[Document]:
        bs4 = _get_bs4()
        if bs4 is None or not html:
            return []

        soup = bs4.BeautifulSoup(html, "html.parser")  # type: ignore[attr-defined]
        tables: List[Document] = []
        for index, table in enumerate(soup.find_all("table")):
            rows = []
            for row in table.find_all("tr"):
                cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
                if cells:
                    rows.append(",".join(cells))
            if not rows:
                continue
            tables.append(
                Document(
                    page_content="\n".join(rows),
                    metadata={
                        "source": file_path,
                        "content_type": "html_table",
                        "table_index": index,
                    },
                )
            )
        return tables

    def _extract_html_images(self, html: str, file_path: str) -> List[Document]:
        bs4 = _get_bs4()
        if bs4 is None or not html:
            return []

        soup = bs4.BeautifulSoup(html, "html.parser")  # type: ignore[attr-defined]
        images: List[Document] = []
        for index, image in enumerate(soup.find_all("img")):
            alt_text = (image.get("alt") or "").strip()
            if not alt_text:
                continue
            images.append(
                Document(
                    page_content=alt_text,
                    metadata={
                        "source": file_path,
                        "content_type": "html_image_alt",
                        "image_index": index,
                    },
                )
            )
        return images
