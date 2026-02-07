"""
Z-Image Turbo Handler - Text to image generation

Alibaba's Z-Image Turbo model for fast image generation.
Optimized for CPU-only systems (no dedicated GPU required).

Requirements:
    pip install diffusers torch accelerate pillow
"""

import os
import hashlib
import logging
import time
from typing import Dict, Any, Optional
from pathlib import Path

from .base import BaseHandler
from ..models.request import Request
from ..models.response import Response, ResponseMeta

logger = logging.getLogger(__name__)

# Lazy imports to avoid loading heavy dependencies at startup
_pipeline = None
_torch = None


def _get_pipeline(config: Dict[str, Any]):
    """Lazy load the Z-Image pipeline."""
    global _pipeline, _torch

    if _pipeline is not None:
        return _pipeline

    try:
        import torch
        _torch = torch

        from diffusers import AutoPipelineForText2Image
        from huggingface_hub import snapshot_download

        model_id = config.get("model_id", "Tongyi-MAI/Z-Image-Turbo")
        device = config.get("device", "cpu")
        dtype = torch.float32 if device == "cpu" else torch.float16

        logger.info(f"Loading Z-Image Turbo model: {model_id}")
        logger.info(f"Device: {device}, dtype: {dtype}")

        # Download model to local directory (avoids symlink issues on Windows)
        local_dir = os.path.join(os.path.expanduser("~"), ".zimage_models", model_id.replace("/", "--"))
        os.makedirs(local_dir, exist_ok=True)

        logger.info(f"Downloading model to: {local_dir}")
        model_path = snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )

        # Load pipeline with CPU optimizations from local path
        _pipeline = AutoPipelineForText2Image.from_pretrained(
            model_path,
            torch_dtype=dtype,
            use_safetensors=True,
            local_files_only=True,
        )

        # Move to device
        _pipeline = _pipeline.to(device)

        # CPU optimizations
        if device == "cpu":
            # Enable memory efficient attention if available
            try:
                _pipeline.enable_attention_slicing(1)
            except:
                pass

            # Enable sequential CPU offload for very low memory
            if config.get("low_memory", False):
                try:
                    _pipeline.enable_sequential_cpu_offload()
                except:
                    pass

        logger.info("Z-Image Turbo model loaded successfully")
        return _pipeline

    except ImportError as e:
        logger.error(f"Missing dependencies for Z-Image: {e}")
        logger.error("Install with: pip install diffusers torch accelerate pillow")
        raise
    except Exception as e:
        logger.error(f"Failed to load Z-Image model: {e}")
        raise


class ZImageHandler(BaseHandler):
    """
    Z-Image Turbo for text-to-image generation.

    Optimized for CPU-only systems with 32GB RAM.
    Generates images from text prompts.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Model settings
        self.model_id = config.get("model_id", "Tongyi-MAI/Z-Image-Turbo")
        self.device = config.get("device", "cpu")

        # Generation settings (optimized for CPU)
        self.default_width = config.get("width", 512)
        self.default_height = config.get("height", 512)
        self.num_inference_steps = config.get("steps", 8)  # Turbo uses fewer steps
        self.guidance_scale = config.get("guidance_scale", 0.0)  # Turbo doesn't need guidance

        # Output settings
        self.output_dir = config.get("output_dir", "output/images")
        self.timeout = config.get("timeout", 300)  # 5 minutes default for CPU

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Lazy loading flag
        self._initialized = False
        self._low_memory = config.get("low_memory", False)

    def _ensure_initialized(self):
        """Ensure pipeline is loaded before first use."""
        if not self._initialized:
            _get_pipeline({
                "model_id": self.model_id,
                "device": self.device,
                "low_memory": self._low_memory,
            })
            self._initialized = True

    def process(self, request: Request) -> Response:
        """Generate image from text prompt."""

        prompt = request.text

        if not prompt:
            return Response.error_response(
                error="No prompt provided",
                code="MISSING_PROMPT"
            )

        # Extract generation parameters from metadata
        metadata = request.metadata or {}
        width = metadata.get("width", self.default_width)
        height = metadata.get("height", self.default_height)
        steps = metadata.get("steps", self.num_inference_steps)
        seed = metadata.get("seed", None)
        negative_prompt = metadata.get("negative_prompt", "")

        # Generate unique output filename
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        timestamp = int(time.time())
        output_filename = f"zimage_{prompt_hash}_{timestamp}.png"
        output_path = os.path.join(self.output_dir, output_filename)

        try:
            # Initialize pipeline (lazy loading)
            self._ensure_initialized()

            logger.info(f"Generating image: {width}x{height}, {steps} steps")
            start_time = time.time()

            # Set seed for reproducibility
            generator = None
            if seed is not None:
                generator = _torch.Generator(device=self.device).manual_seed(seed)

            # Generate image
            result = _pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=self.guidance_scale,
                generator=generator,
            )

            # Save image
            image = result.images[0]
            image.save(output_path)

            elapsed = time.time() - start_time
            logger.info(f"Image generated in {elapsed:.1f}s: {output_path}")

            return Response(
                success=True,
                image_path=output_path,
                text=f"Generated image from: {prompt[:50]}...",
                output={
                    "image_path": output_path,
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "seed": seed,
                    "generation_time_seconds": round(elapsed, 2),
                    "model": self.model_id,
                },
                meta=ResponseMeta()
            )

        except ImportError as e:
            return Response.error_response(
                error=f"Z-Image dependencies not installed: {e}",
                code="DEPENDENCY_MISSING"
            )
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return Response.error_response(
                error=f"Image generation failed: {str(e)}",
                code="GENERATION_ERROR"
            )

    def is_available(self) -> bool:
        """Check if Z-Image dependencies are available."""
        try:
            import torch
            import diffusers
            from PIL import Image
            return True
        except ImportError:
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get handler status."""
        available = self.is_available()
        initialized = self._initialized

        status = {
            "handler": "ZImageHandler",
            "model_id": self.model_id,
            "device": self.device,
            "available": available,
            "initialized": initialized,
            "default_size": f"{self.default_width}x{self.default_height}",
            "default_steps": self.num_inference_steps,
        }

        if available and _torch is not None:
            status["torch_version"] = _torch.__version__
            if self.device == "cpu":
                status["cpu_threads"] = _torch.get_num_threads()

        return status


class ZImageHTTPHandler(BaseHandler):
    """
    Z-Image via HTTP API (fal.ai, getimg.ai, etc.)

    Use this for faster generation without local GPU.
    Requires API key configuration.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.api_url = config.get("api_url", "https://fal.ai/models/fal-ai/z-image/turbo")
        self.api_key = config.get("api_key", os.environ.get("ZIMAGE_API_KEY", ""))
        self.timeout = config.get("timeout", 60)
        self.output_dir = config.get("output_dir", "output/images")

        os.makedirs(self.output_dir, exist_ok=True)

    def process(self, request: Request) -> Response:
        """Generate image via HTTP API."""
        import requests

        prompt = request.text
        if not prompt:
            return Response.error_response(
                error="No prompt provided",
                code="MISSING_PROMPT"
            )

        if not self.api_key:
            return Response.error_response(
                error="Z-Image API key not configured",
                code="NO_API_KEY"
            )

        try:
            # Call API (example for fal.ai format)
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "prompt": prompt,
                    "image_size": "square",
                    "num_inference_steps": 8,
                },
                timeout=self.timeout,
            )

            if response.status_code == 200:
                result = response.json()
                image_url = result.get("images", [{}])[0].get("url", "")

                return Response(
                    success=True,
                    text=f"Generated image from: {prompt[:50]}...",
                    output={
                        "image_url": image_url,
                        "prompt": prompt,
                        "source": "api",
                    },
                    meta=ResponseMeta()
                )
            else:
                return Response.error_response(
                    error=f"API error: {response.status_code}",
                    code="API_ERROR"
                )

        except requests.Timeout:
            return Response.error_response(
                error="API timeout",
                code="TIMEOUT"
            )
        except Exception as e:
            return Response.error_response(
                error=f"API request failed: {e}",
                code="REQUEST_ERROR"
            )

    def is_available(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key)
