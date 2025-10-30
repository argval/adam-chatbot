import io
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .document_processor import DocumentProcessor


logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


@dataclass(frozen=True)
class PDFSplitConfig:
    semantic_threshold: int = int(os.getenv("PDF_SEMANTIC_THRESHOLD", 5000))
    similarity_cutoff: float = float(os.getenv("PDF_SIMILARITY_CUTOFF", 0.7))
    min_sentence_length: int = int(os.getenv("PDF_MIN_SENTENCE_LENGTH", 24))
    enable_ocr: bool = os.getenv("PDF_ENABLE_OCR", "false").lower() in {"1", "true", "yes", "on"}
    enable_table_extraction: bool = (
        os.getenv("PDF_ENABLE_TABLES", "false").lower() in {"1", "true", "yes", "on"}
    )


@lru_cache(maxsize=1)
def _load_semantic_model(model_name: str = DEFAULT_MODEL_NAME):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class PDFProcessor(DocumentProcessor):
    def process(
        self, file_path: str, chunk_size: int, chunk_overlap: int
    ) -> Dict[str, List[Document]]:
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()

            if not pages:
                logging.warning(f"PyPDFLoader returned no pages for {file_path}.")
                return {"chunks": [], "classes": []}

            character_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )

            config = PDFSplitConfig()
            full_text = "\n".join([page.page_content for page in pages])
            if len(full_text) > config.semantic_threshold:
                chunks = self._semantic_split(full_text, file_path, config)
                if not chunks:
                    chunks = character_splitter.split_documents(pages)
            else:
                chunks = character_splitter.split_documents(pages)

            extras: List[Document] = []
            if config.enable_table_extraction:
                extras.extend(self._extract_tables(file_path))
            if config.enable_ocr:
                extras.extend(self._extract_image_texts(file_path))

            for doc in extras:
                doc.metadata.setdefault("source", file_path)
                doc.metadata.setdefault("source_name", Path(file_path).name)
            chunks.extend(extras)

            for chunk in chunks:
                chunk.metadata["source"] = file_path
                chunk.metadata["source_name"] = Path(file_path).name

            return {"chunks": chunks, "classes": []}
        except Exception as e:
            logging.error(f"Failed to process PDF {file_path}. Error: {e}")
            return {"chunks": [], "classes": []}

    def _semantic_split(
        self, text: str, file_path: str, config: PDFSplitConfig
    ) -> List[Document]:
        sentences = [
            sentence.strip()
            for sentence in re_split_sentences(text)
            if len(sentence.strip()) >= config.min_sentence_length
        ]
        if len(sentences) < 2:
            return []

        model = _load_semantic_model()
        embeddings = model.encode(sentences, normalize_embeddings=True)
        chunks: List[Document] = []
        current_chunk = sentences[0]

        for idx in range(1, len(sentences)):
            similarity = float(np.dot(embeddings[idx - 1], embeddings[idx]))
            if similarity >= config.similarity_cutoff:
                current_chunk += " " + sentences[idx]
            else:
                chunks.append(
                    Document(
                        page_content=current_chunk,
                        metadata={"source": file_path},
                    )
                )
                current_chunk = sentences[idx]

        chunks.append(
            Document(page_content=current_chunk, metadata={"source": file_path})
        )
        return chunks

    def _extract_tables(self, file_path: str) -> List[Document]:
        pdfplumber = _pdfplumber_available()
        if pdfplumber is None:
            return []

        tables: List[Document] = []
        try:
            with pdfplumber.open(file_path) as pdf:  # type: ignore[attr-defined]
                for page_index, page in enumerate(pdf.pages, start=1):
                    try:
                        extracted = page.extract_tables()
                    except Exception as exc:  # pragma: no cover
                        logger.debug("Failed to extract tables from %s page %s: %s", file_path, page_index, exc)
                        continue
                    for table_index, table in enumerate(extracted):
                        if not table:
                            continue
                        table_lines = [",".join(cell or "" for cell in row) for row in table]
                        table_text = "\n".join(table_lines)
                        tables.append(
                            Document(
                                page_content=table_text,
                                metadata={
                                    "source": file_path,
                                    "source_name": Path(file_path).name,
                                    "content_type": "table",
                                    "source_page": page_index,
                                    "table_index": table_index,
                                },
                            )
                        )
        except Exception as exc:  # pragma: no cover
            logger.debug("pdfplumber failed for %s: %s", file_path, exc)
        return tables

    def _extract_image_texts(self, file_path: str) -> List[Document]:
        fitz = _fitz_available()
        pytesseract = _pytesseract_available()
        Image = _pil_image_available()
        if None in (fitz, pytesseract, Image):
            return []

        image_docs: List[Document] = []
        try:
            with fitz.open(file_path) as pdf:  # type: ignore[attr-defined]
                for page_index, page in enumerate(pdf, start=1):
                    for image_index, img in enumerate(page.get_images(full=True)):
                        xref = img[0]
                        try:
                            base_image = pdf.extract_image(xref)
                            image_bytes = base_image.get("image")
                            if image_bytes is None:
                                continue
                            pil_image = Image.open(io.BytesIO(image_bytes))  # type: ignore[arg-type]
                            text = pytesseract.image_to_string(pil_image)
                        except Exception as exc:  # pragma: no cover
                            logger.debug(
                                "Failed OCR for %s image %s on page %s: %s",
                                file_path,
                                image_index,
                                page_index,
                                exc,
                            )
                            continue

                        text = text.strip()
                        if not text:
                            continue
                        image_docs.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": file_path,
                                    "source_name": Path(file_path).name,
                                    "content_type": "image_ocr",
                                    "source_page": page_index,
                                    "image_index": image_index,
                                },
                            )
                        )
        except Exception as exc:  # pragma: no cover
            logger.debug("PyMuPDF failed for %s: %s", file_path, exc)

        return image_docs


    
    
    
def _try_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None


class _ImportFlags:
    pdfplumber_warning_emitted = False
    ocr_warning_emitted = False

    
    
    
def _pdfplumber_available():
    module = _try_import("pdfplumber")
    if module is None and not _ImportFlags.pdfplumber_warning_emitted:
        logger.warning(
            "pdfplumber is not installed; table extraction from PDFs is disabled."
        )
        _ImportFlags.pdfplumber_warning_emitted = True
    return module


def _fitz_available():
    module = _try_import("fitz")
    if module is None and not _ImportFlags.ocr_warning_emitted:
        logger.warning(
            "PyMuPDF (fitz) is not installed; image OCR for PDFs is disabled."
        )
        _ImportFlags.ocr_warning_emitted = True
    return module


def _pytesseract_available():
    module = _try_import("pytesseract")
    if module is None and not _ImportFlags.ocr_warning_emitted:
        logger.warning(
            "pytesseract is not installed; image OCR for PDFs is disabled."
        )
        _ImportFlags.ocr_warning_emitted = True
    return module


def _pil_image_available():
    try:
        from PIL import Image  # type: ignore

        return Image
    except ImportError:
        if not _ImportFlags.ocr_warning_emitted:
            logger.warning(
                "Pillow is not installed; image OCR for PDFs is disabled."
            )
            _ImportFlags.ocr_warning_emitted = True
        return None


def re_split_sentences(text: str) -> List[str]:


def re_split_sentences(text: str) -> List[str]:
    import re

    return re.split(r"(?<=[.!?])\s+", text)

    
    
    
