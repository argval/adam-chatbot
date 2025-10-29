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

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


@dataclass(frozen=True)
class PDFSplitConfig:
    semantic_threshold: int = int(os.getenv("PDF_SEMANTIC_THRESHOLD", 5000))
    similarity_cutoff: float = float(os.getenv("PDF_SIMILARITY_CUTOFF", 0.7))
    min_sentence_length: int = int(os.getenv("PDF_MIN_SENTENCE_LENGTH", 24))


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


def re_split_sentences(text: str) -> List[str]:
    import re

    return re.split(r"(?<=[.!?])\s+", text)
