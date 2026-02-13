"""
Vision Service - Image understanding via SmolVLM2 (Ollama).

Uses SmolVLM2 2.2B through Ollama's multimodal API to:
- Describe image contents
- Answer questions about images
- Extract text from images (basic OCR)

DESIGN:
- Uses Ollama /api/generate with images parameter
- Lazy: only calls API when image is actually present
- Separate model from main LLM (SmolVLM2 for vision, Qwen3 for text)
- NOT a handler: preprocessing service injected into MistralHandler
"""

import base64
import logging
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class VisionResult:
    """Result of vision analysis."""
    success: bool
    description: str = ""
    elapsed_ms: int = 0
    error: Optional[str] = None
    model: str = ""


class VisionService:
    """
    Image understanding via SmolVLM2 through Ollama.

    NOT a handler. This is a preprocessing service called by handlers
    to generate image descriptions before LLM processing.
    """

    def __init__(self, config: Dict[str, Any]):
        self.enabled = config.get("enabled", True)
        self.model = config.get("model", "richardyoung/smolvlm2-2.2b-instruct")
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.timeout = config.get("timeout", 120)
        self.max_image_size = config.get("max_image_size", 10_000_000)  # 10MB
        self._available = None

    def is_available(self) -> bool:
        """Check if vision model is available in Ollama."""
        if self._available is not None:
            return self._available

        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                model_names = [m.get("name", "") for m in models]
                # Check both full name and short name
                self._available = any(
                    self.model in name or name.startswith(self.model.split("/")[-1])
                    for name in model_names
                )
            else:
                self._available = False
        except Exception:
            self._available = False

        logger.info(f"VisionService available: {self._available} (model: {self.model})")
        return self._available

    def analyze_image(self, image_data: str, question: str = "") -> VisionResult:
        """
        Analyze an image using SmolVLM2.

        Args:
            image_data: Base64-encoded image data (may include data URL prefix)
            question: Optional question about the image

        Returns:
            VisionResult with description
        """
        if not self.enabled:
            return VisionResult(success=False, error="Vision service disabled")

        if not self.is_available():
            return VisionResult(success=False, error=f"Vision model {self.model} not available in Ollama")

        start = time.time()

        try:
            # Clean base64 data (remove data URL prefix if present)
            clean_data = image_data
            if clean_data.startswith('data:'):
                clean_data = clean_data.split(',', 1)[1] if ',' in clean_data else clean_data

            # Check size
            try:
                raw_size = len(base64.b64decode(clean_data))
                if raw_size > self.max_image_size:
                    return VisionResult(
                        success=False,
                        error=f"Image too large: {raw_size / 1024 / 1024:.1f}MB (max {self.max_image_size / 1024 / 1024:.0f}MB)"
                    )
            except Exception:
                pass  # If we can't decode to check size, let Ollama handle it

            # Build prompt
            prompt = question if question else "Describe this image in detail. What do you see?"

            # Call Ollama multimodal API
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "images": [clean_data],
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 512,
                    },
                },
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()
            description = result.get("response", "").strip()
            elapsed = int((time.time() - start) * 1000)

            if not description:
                return VisionResult(
                    success=False,
                    error="Vision model returned empty response",
                    elapsed_ms=elapsed,
                    model=self.model,
                )

            return VisionResult(
                success=True,
                description=description,
                elapsed_ms=elapsed,
                model=self.model,
            )

        except requests.Timeout:
            elapsed = int((time.time() - start) * 1000)
            return VisionResult(
                success=False,
                error=f"Vision analysis timed out after {self.timeout}s",
                elapsed_ms=elapsed,
                model=self.model,
            )

        except requests.ConnectionError:
            return VisionResult(
                success=False,
                error=f"Cannot connect to Ollama at {self.base_url}",
                model=self.model,
            )

        except Exception as e:
            elapsed = int((time.time() - start) * 1000)
            return VisionResult(
                success=False,
                error=f"Vision analysis error: {str(e)}",
                elapsed_ms=elapsed,
                model=self.model,
            )

    def analyze_attachment(self, attachment) -> VisionResult:
        """
        Analyze an image attachment object.

        Args:
            attachment: Attachment object with .data, .name, .type attributes

        Returns:
            VisionResult with description
        """
        data = getattr(attachment, 'data', '')
        name = getattr(attachment, 'name', 'image')

        if not data:
            return VisionResult(success=False, error=f"No image data in attachment {name}")

        # Use filename as context hint
        question = f"Describe this image ({name}) in detail. What do you see? If there is text, read it."

        return self.analyze_image(data, question)
