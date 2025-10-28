from .document_processor import DocumentProcessor
from .pdf_processor import PDFProcessor
from .text_processor import TextProcessor


class DocumentProcessorFactory:
    """
    A stateless factory that returns the appropriate DocumentProcessor
    strategy for a given file path.
    """

    def __init__(self):
        self._processors = {
            # PDF files
            "pdf": PDFProcessor(),
            # Text files
            "txt": TextProcessor(),
            "md": TextProcessor(),
            "html": TextProcessor(),
            "docx": TextProcessor(),
        }

    def get_processor(self, file_path: str) -> DocumentProcessor | None:
        """
        Returns the appropriate processor for the given file path based on its extension.
        """
        extension = file_path.split(".")[-1].lower()
        return self._processors.get(extension)
