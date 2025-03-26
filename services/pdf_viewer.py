from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class PDFViewerMetadata:
    pdf_path: str
    page_no: str
    title: str
    width: int = 595  # A4 기준
    height: int = 842


class PDFViewerManager:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def get_page_metadata(self, page_no: str) -> PDFViewerMetadata:
        return PDFViewerMetadata(
            pdf_path=self.pdf_path, page_no=page_no, title="뉴런과학1_미니북"
        )
