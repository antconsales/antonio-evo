"""
Document Parser Service - OCR/VLM-based document understanding.

Uses dots.ocr (1.7B VLM) to extract structured text from:
- Images (PNG, JPEG, WEBP, BMP, TIFF)
- PDFs (converted to images first via pymupdf)

DESIGN PRINCIPLES:
- Lazy model loading (only loads on first document)
- Graceful fallback (if dots.ocr unavailable, returns None)
- NEVER executes attachment content
- Configurable via environment + handlers.json
"""

import base64
import io
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DocumentParseResult:
    """Result of parsing a document."""
    success: bool
    text: str = ""
    elements: List[Dict] = field(default_factory=list)
    page_count: int = 1
    elapsed_ms: int = 0
    error: Optional[str] = None
    source_type: str = ""  # "image" or "pdf"


class DocumentParser:
    """
    OCR/VLM service for document understanding.

    NOT a handler. This is a preprocessing service called by handlers
    to enrich attachment context before LLM processing.

    Supports two modes:
    - Local: loads dots.ocr model via transformers (GPU recommended)
    - HTTP: queries a vLLM server running dots.ocr
    """

    def __init__(self, config: Dict[str, Any]):
        self.enabled = config.get("enabled", True)
        self.model_id = config.get("model_id", "yifeihu/dots.llm.ocr.1.7B")
        self.device = config.get("device", "cpu")
        self.max_pages = config.get("max_pages", 5)
        self.max_image_size = config.get("max_image_size", 10_000_000)  # 10MB
        self.timeout_seconds = config.get("timeout_seconds", 60)
        self.use_http = config.get("use_http", False)
        self.http_url = config.get("http_url", "http://localhost:8805")

        # Lazy loading state
        self._model = None
        self._processor = None
        self._loaded = False
        self._load_error: Optional[str] = None

        # Stats
        self._parse_count = 0
        self._total_ms = 0

    def is_available(self) -> bool:
        """Check if document parsing dependencies are installed."""
        if not self.enabled:
            return False

        if self.use_http:
            return self._check_http_server()

        try:
            import torch  # noqa: F401
            from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: F401
            return True
        except ImportError:
            return False

    def parse_attachment(self, attachment) -> Optional[DocumentParseResult]:
        """
        Parse a single attachment. Returns None if not parseable.

        SECURITY: attachment.data is base64-encoded UNTRUSTED INERT DATA.
        We decode it only to pass as pixel data to the VLM. Never executed.
        """
        if not self.enabled:
            return None

        is_image = getattr(attachment, 'is_image', lambda: False)()
        is_pdf = getattr(attachment, 'is_pdf', lambda: False)()

        if not is_image and not is_pdf:
            return None

        att_size = getattr(attachment, 'size', 0)
        if att_size > self.max_image_size:
            return DocumentParseResult(
                success=False,
                error=f"Attachment too large ({att_size} bytes, max {self.max_image_size})",
                source_type="image" if is_image else "pdf",
            )

        # Decode base64
        try:
            data_str = getattr(attachment, 'data', '')
            if data_str.startswith('data:'):
                data_str = data_str.split(',', 1)[1] if ',' in data_str else data_str
            raw_bytes = base64.b64decode(data_str)
        except Exception as e:
            return DocumentParseResult(
                success=False,
                error=f"Base64 decode failed: {e}",
            )

        if is_pdf:
            return self.parse_pdf(raw_bytes)
        else:
            mime_type = getattr(attachment, 'type', 'image/png')
            return self.parse_image(raw_bytes, mime_type)

    def parse_image(self, image_data: bytes, mime_type: str = "image/png") -> DocumentParseResult:
        """Parse a single image and extract text."""
        start = time.time()

        try:
            from PIL import Image
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception as e:
            return DocumentParseResult(
                success=False,
                error=f"Cannot open image: {e}",
                source_type="image",
            )

        if self.use_http:
            result = self._parse_via_http(image)
        else:
            result = self._parse_via_local(image)

        result.source_type = "image"
        result.elapsed_ms = int((time.time() - start) * 1000)
        self._parse_count += 1
        self._total_ms += result.elapsed_ms
        return result

    def parse_pdf(self, pdf_data: bytes) -> DocumentParseResult:
        """Parse a PDF document by converting pages to images and OCR-ing each."""
        start = time.time()

        try:
            images = self._pdf_to_images(pdf_data)
        except ImportError:
            return DocumentParseResult(
                success=False,
                error="PDF support requires pymupdf: pip install pymupdf",
                source_type="pdf",
            )
        except Exception as e:
            return DocumentParseResult(
                success=False,
                error=f"PDF conversion failed: {e}",
                source_type="pdf",
            )

        if not images:
            return DocumentParseResult(
                success=False,
                error="PDF has no pages",
                source_type="pdf",
            )

        all_text = []
        all_elements = []

        for i, img in enumerate(images):
            if self.use_http:
                page_result = self._parse_via_http(img)
            else:
                page_result = self._parse_via_local(img)

            if page_result.success and page_result.text:
                all_text.append(f"--- Page {i + 1} ---\n{page_result.text}")
                all_elements.extend(page_result.elements)

        elapsed = int((time.time() - start) * 1000)
        self._parse_count += 1
        self._total_ms += elapsed

        if not all_text:
            return DocumentParseResult(
                success=False,
                error="No text extracted from any page",
                source_type="pdf",
                page_count=len(images),
                elapsed_ms=elapsed,
            )

        return DocumentParseResult(
            success=True,
            text="\n\n".join(all_text),
            elements=all_elements,
            page_count=len(images),
            elapsed_ms=elapsed,
            source_type="pdf",
        )

    def _lazy_load_model(self) -> bool:
        """Load dots.ocr model on first use."""
        if self._loaded:
            return self._model is not None

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            logger.info(f"Loading dots.ocr model: {self.model_id} on {self.device}...")

            self._processor = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )

            dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                device_map=self.device if self.device != "cpu" else None,
                trust_remote_code=True,
            )
            if self.device == "cpu":
                self._model = self._model.to("cpu")

            self._loaded = True
            logger.info("dots.ocr model loaded successfully")
            return True

        except ImportError as e:
            self._load_error = f"Missing dependencies: {e}"
            self._loaded = True
            logger.warning(f"dots.ocr not available: {self._load_error}")
            return False
        except Exception as e:
            self._load_error = f"Model load failed: {e}"
            self._loaded = True
            logger.warning(f"dots.ocr load error: {self._load_error}")
            return False

    def _parse_via_local(self, image) -> DocumentParseResult:
        """Parse image using locally loaded dots.ocr model."""
        if not self._lazy_load_model():
            return DocumentParseResult(
                success=False,
                error=f"Model not available: {self._load_error}",
            )

        try:
            from qwen_vl_utils import process_vision_info

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Parse this document. Extract all text, tables, and formulas."},
                    ],
                }
            ]

            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            if self.device != "cpu":
                inputs = inputs.to(self.device)

            generated_ids = self._model.generate(**inputs, max_new_tokens=8192)
            output_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            formatted = self._format_ocr_output(output_text)
            return DocumentParseResult(
                success=True,
                text=formatted,
                elements=self._try_parse_elements(output_text),
            )

        except ImportError as e:
            return DocumentParseResult(
                success=False,
                error=f"Missing dependency: {e}",
            )
        except Exception as e:
            logger.error(f"dots.ocr inference error: {e}")
            return DocumentParseResult(
                success=False,
                error=f"OCR failed: {e}",
            )

    def _parse_via_http(self, image) -> DocumentParseResult:
        """Parse image via vLLM HTTP server running dots.ocr."""
        try:
            import requests
            from PIL import Image

            # Convert PIL image to base64
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            response = requests.post(
                f"{self.http_url}/v1/chat/completions",
                json={
                    "model": self.model_id,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                                },
                                {
                                    "type": "text",
                                    "text": "Parse this document. Extract all text, tables, and formulas.",
                                },
                            ],
                        }
                    ],
                    "max_tokens": 8192,
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()

            result = response.json()
            output_text = result["choices"][0]["message"]["content"]

            formatted = self._format_ocr_output(output_text)
            return DocumentParseResult(
                success=True,
                text=formatted,
                elements=self._try_parse_elements(output_text),
            )

        except Exception as e:
            return DocumentParseResult(
                success=False,
                error=f"HTTP OCR failed: {e}",
            )

    def _pdf_to_images(self, pdf_data: bytes) -> list:
        """Convert PDF pages to PIL Images."""
        import fitz  # pymupdf
        from PIL import Image

        images = []
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        page_count = min(len(doc), self.max_pages)

        for i in range(page_count):
            page = doc[i]
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)

        doc.close()
        return images

    def _format_ocr_output(self, raw_output: str) -> str:
        """Convert dots.ocr output to readable markdown."""
        try:
            elements = json.loads(raw_output)
            if not isinstance(elements, list):
                return raw_output.strip()

            parts = []
            for elem in elements:
                category = elem.get("category", "Text")
                text = elem.get("text", "").strip()
                if not text:
                    continue

                if category in ("Title", "Section-header"):
                    parts.append(f"## {text}")
                elif category == "Table":
                    parts.append(text)  # Already HTML
                elif category == "Formula":
                    parts.append(f"$${text}$$")
                elif category in ("Caption", "Footnote"):
                    parts.append(f"*{text}*")
                else:
                    parts.append(text)

            return "\n\n".join(parts) if parts else raw_output.strip()

        except (json.JSONDecodeError, TypeError):
            # Model returned plain text, use as-is
            return raw_output.strip()

    def _try_parse_elements(self, raw_output: str) -> List[Dict]:
        """Try to parse raw output as JSON elements list."""
        try:
            elements = json.loads(raw_output)
            if isinstance(elements, list):
                return elements
        except (json.JSONDecodeError, TypeError):
            pass
        return []

    def _check_http_server(self) -> bool:
        """Check if the HTTP OCR server is reachable."""
        try:
            import requests
            resp = requests.get(f"{self.http_url}/health", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Return parser statistics."""
        return {
            "enabled": self.enabled,
            "model_id": self.model_id,
            "device": self.device,
            "use_http": self.use_http,
            "model_loaded": self._loaded and self._model is not None,
            "load_error": self._load_error,
            "parse_count": self._parse_count,
            "total_ms": self._total_ms,
            "avg_ms": self._total_ms // max(self._parse_count, 1),
        }
