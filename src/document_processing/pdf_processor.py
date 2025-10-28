import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .document_processor import DocumentProcessor

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


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

            # Hybrid for PDFs: Semantic for long docs using sentence-transformers, character for short
            from sklearn.metrics.pairwise import cosine_similarity

            character_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )

            # Hybrid logic: Use semantic if document is long (>5000 chars)
            full_text = "\n".join([page.page_content for page in pages])
            if len(full_text) > 5000:
                # Semantic splitting using sentence-transformers
                model = _load_semantic_model()
                sentences = [sentence.strip() for sentence in full_text.split(". ") if sentence.strip()]
                if not sentences:
                    chunks = character_splitter.split_documents(pages)
                else:
                    embeddings = model.encode(sentences)
                    similarities = cosine_similarity(embeddings)
                    chunks = []
                    current_chunk = sentences[0]
                    for i in range(1, len(sentences)):
                        if similarities[i - 1, i] > 0.7:
                            current_chunk += ". " + sentences[i]
                        else:
                            chunks.append(Document(page_content=current_chunk, metadata={"source": file_path}))
                            current_chunk = sentences[i]
                    chunks.append(Document(page_content=current_chunk, metadata={"source": file_path}))
            else:
                chunks = character_splitter.split_documents(pages)

            for chunk in chunks:
                chunk.metadata["source"] = file_path
                chunk.metadata["source_name"] = Path(file_path).name

            return {"chunks": chunks, "classes": []}
        except Exception as e:
            logging.error(f"Failed to process PDF {file_path}. Error: {e}")
            return {"chunks": [], "classes": []}
