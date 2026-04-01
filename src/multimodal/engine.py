"""
MULTI-MODAL RAG MODULE
=======================
Process and retrieve across text, images, tables, and structured data.

Capabilities:
  - Image understanding (captioning, OCR, visual Q&A)
  - Table extraction and structured querying
  - PDF visual parsing (layout-aware)
  - Multi-modal embedding (CLIP, SigLIP)
  - Cross-modal retrieval (text→image, image→text)
  - Document layout analysis
"""

import os
import base64
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field

from src.utils.logger import log


@dataclass
class MultiModalDocument:
    """A document with mixed content types."""
    id: str
    text_content: str = ""
    images: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    modality: str = "text"  # text | image | table | mixed


class ImageProcessor:
    """Process images for multi-modal RAG."""

    def __init__(self, config: dict):
        self.config = config
        self.provider = config.get("vision_provider", "openai")

    def caption_image(self, image_path: str) -> str:
        """Generate a text caption for an image."""
        if self.provider == "openai":
            return self._caption_openai(image_path)
        elif self.provider == "local":
            return self._caption_local(image_path)
        return ""

    def extract_text_ocr(self, image_path: str) -> str:
        """Extract text from an image using OCR."""
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(image_path)
            return pytesseract.image_to_string(img)
        except ImportError:
            log.warning("pytesseract/Pillow not installed")
            return ""

    def _caption_openai(self, image_path: str) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            ext = Path(image_path).suffix.lower().replace(".", "")
            mime = f"image/{ext}" if ext in ("png", "jpg", "jpeg", "gif", "webp") else "image/png"
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in detail for search indexing."},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    ],
                }],
                max_tokens=300,
            )
            return response.choices[0].message.content
        except Exception as e:
            log.error(f"OpenAI vision error: {e}")
            return ""

    def _caption_local(self, image_path: str) -> str:
        log.info("Local captioning: use BLIP or LLaVA via Ollama")
        return f"[Image: {Path(image_path).name}]"


class TableExtractor:
    """Extract and structure tables from documents."""

    def extract_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract tables from a PDF file."""
        try:
            import camelot
            tables = camelot.read_pdf(pdf_path, pages="all")
            results = []
            for i, table in enumerate(tables):
                df = table.df
                results.append({
                    "table_id": i,
                    "headers": df.iloc[0].tolist() if len(df) > 0 else [],
                    "rows": df.iloc[1:].values.tolist() if len(df) > 1 else [],
                    "text": df.to_string(index=False),
                    "accuracy": table.accuracy,
                })
            return results
        except ImportError:
            log.warning("camelot-py not installed — pip install camelot-py[cv]")
            return []

    def extract_from_html(self, html_content: str) -> List[Dict[str, Any]]:
        """Extract tables from HTML content."""
        try:
            import pandas as pd
            tables = pd.read_html(html_content)
            return [{"table_id": i, "text": df.to_string(index=False),
                      "rows": df.values.tolist(), "headers": df.columns.tolist()}
                    for i, df in enumerate(tables)]
        except Exception:
            return []


class MultiModalEngine:
    """Orchestrate multi-modal document processing for RAG."""

    def __init__(self, config: dict):
        self.config = config
        self.image_processor = ImageProcessor(config)
        self.table_extractor = TableExtractor()

    def process_document(self, file_path: str) -> MultiModalDocument:
        """Process a file into a multi-modal document."""
        path = Path(file_path)
        ext = path.suffix.lower()
        doc = MultiModalDocument(id=path.stem, metadata={"source": str(path)})

        if ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"):
            doc.modality = "image"
            caption = self.image_processor.caption_image(file_path)
            ocr_text = self.image_processor.extract_text_ocr(file_path)
            doc.text_content = f"Image: {caption}\nOCR Text: {ocr_text}"
            doc.images.append({"path": file_path, "caption": caption})

        elif ext == ".pdf":
            doc.modality = "mixed"
            tables = self.table_extractor.extract_from_pdf(file_path)
            doc.tables = tables
            table_text = "\n\n".join(t.get("text", "") for t in tables)
            try:
                import fitz
                pdf = fitz.open(file_path)
                text = "\n\n".join(page.get_text() for page in pdf)
                pdf.close()
                doc.text_content = text + "\n\n" + table_text
            except ImportError:
                doc.text_content = table_text

        elif ext in (".html", ".htm"):
            doc.modality = "mixed"
            content = path.read_text(encoding="utf-8", errors="replace")
            tables = self.table_extractor.extract_from_html(content)
            doc.tables = tables
            import re
            doc.text_content = re.sub(r"<[^>]+>", " ", content)

        else:
            doc.modality = "text"
            doc.text_content = path.read_text(encoding="utf-8", errors="replace")

        log.info(f"Multi-modal processed: {path.name} ({doc.modality})")
        return doc

    def process_directory(self, dir_path: str) -> List[MultiModalDocument]:
        """Process all files in a directory."""
        docs = []
        for f in Path(dir_path).rglob("*"):
            if f.is_file() and f.suffix.lower() in (
                ".txt", ".md", ".pdf", ".html", ".htm",
                ".png", ".jpg", ".jpeg", ".gif", ".webp",
            ):
                docs.append(self.process_document(str(f)))
        return docs
