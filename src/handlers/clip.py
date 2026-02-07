"""
CLIP Handler - Image understanding

Uses CLIP for:
- Image captioning
- Image classification
- Visual question answering (basic)
"""

from typing import Dict, Any, Optional

from .base import BaseHandler
from ..models.request import Request
from ..models.response import Response, ResponseMeta


class CLIPHandler(BaseHandler):
    """
    CLIP for image understanding.

    NOTE: This handler is optional and requires torch + transformers.
    If not available, it returns an error gracefully.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get("model", "ViT-B/32")
        self.enabled = config.get("enabled", False)
        self._model = None
        self._processor = None

    def _load_model(self):
        """Lazy load CLIP model."""
        if self._model is not None:
            return True

        try:
            from transformers import CLIPProcessor, CLIPModel

            self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            return True
        except ImportError:
            return False
        except Exception:
            return False

    def process(self, request: Request) -> Response:
        """Process image for understanding."""

        if not self.enabled:
            return Response.error_response(
                error="CLIP handler is disabled in config",
                code="HANDLER_DISABLED"
            )

        image_path = request.image_path
        image_bytes = request.image_bytes

        if not image_path and not image_bytes:
            return Response.error_response(
                error="No image provided",
                code="MISSING_IMAGE"
            )

        # Load model on first use
        if not self._load_model():
            return Response.error_response(
                error="CLIP model not available. Install: pip install transformers torch",
                code="MODEL_NOT_AVAILABLE"
            )

        try:
            from PIL import Image
            import io

            # Load image
            if image_bytes:
                image = Image.open(io.BytesIO(image_bytes))
            else:
                image = Image.open(image_path)

            # Generate caption using CLIP
            # This is a simplified approach - real caption generation would need BLIP
            candidate_labels = [
                "a photo of a person",
                "a photo of an animal",
                "a photo of food",
                "a photo of a building",
                "a photo of nature",
                "a photo of a vehicle",
                "a photo of text or document",
                "a photo of an object"
            ]

            inputs = self._processor(
                text=candidate_labels,
                images=image,
                return_tensors="pt",
                padding=True
            )

            outputs = self._model(**inputs)
            logits = outputs.logits_per_image
            probs = logits.softmax(dim=1)

            # Get top prediction
            top_idx = probs.argmax().item()
            top_label = candidate_labels[top_idx]
            confidence = probs[0][top_idx].item()

            return Response(
                success=True,
                text=top_label,
                output={
                    "caption": top_label,
                    "confidence": round(confidence, 3),
                    "all_scores": {
                        label: round(probs[0][i].item(), 3)
                        for i, label in enumerate(candidate_labels)
                    }
                },
                meta=ResponseMeta()
            )

        except ImportError:
            return Response.error_response(
                error="PIL not available. Install: pip install Pillow",
                code="DEPENDENCY_MISSING"
            )
        except Exception as e:
            return Response.error_response(
                error=f"Image processing error: {str(e)}",
                code="PROCESSING_ERROR"
            )

    def is_available(self) -> bool:
        """Check if CLIP dependencies are available."""
        if not self.enabled:
            return False

        try:
            import transformers
            import torch
            from PIL import Image
            return True
        except ImportError:
            return False
