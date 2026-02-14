"""Image analysis tool - wraps existing VisionService."""

import base64
from .registry import ToolResult

DEFINITION = {
    "name": "analyze_image",
    "description": (
        "Analyze an image file and describe its contents. "
        "Can also answer specific questions about the image."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Absolute path to the image file",
            },
            "question": {
                "type": "string",
                "description": "Optional question about the image (default: general description)",
            },
        },
        "required": ["image_path"],
    },
}


def create_handler(vision_service):
    """Create image analysis tool handler bound to VisionService instance."""

    def analyze_image(image_path: str, question: str = "") -> ToolResult:
        if not vision_service or not vision_service.is_available():
            return ToolResult(
                success=False,
                output="Image analysis not available (VisionService/Ollama not running).",
            )
        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("ascii")

            # Build prompt
            prompt = question if question else "Describe this image in detail. What do you see?"

            result = vision_service.analyze_image(image_data, prompt)
            if result and result.success:
                return ToolResult(
                    success=True,
                    output=result.description or "Image analyzed but no description returned.",
                    elapsed_ms=result.elapsed_ms,
                )
            error_msg = getattr(result, "error", "Unknown vision error") if result else "VisionService returned None"
            return ToolResult(success=False, output=f"Vision analysis failed: {error_msg}")
        except FileNotFoundError:
            return ToolResult(success=False, output=f"Image not found: {image_path}")
        except Exception as e:
            return ToolResult(success=False, output=f"Error analyzing image: {e}")

    return analyze_image
