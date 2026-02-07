"""
External Handler - Claude/GPT API fallback

USE SPARINGLY.

RULES:
1. ONLY called when policy explicitly allows
2. Every call is logged with justification
3. Cost tracking required
"""

import os
import json
import time
import base64
from typing import Dict, Any, List

from .base import BaseHandler
from ..models.request import Request
from ..models.response import Response, ResponseMeta


class ExternalHandler(BaseHandler):
    """
    External API handler (Claude/GPT).

    This is the LAST RESORT.
    Every call must be justified and logged.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.provider = config.get("provider", "anthropic")  # or "openai"
        self.model = config.get("model", "claude-3-haiku-20240307")
        self.api_key = os.environ.get("EXTERNAL_API_KEY", "")
        self.log_path = config.get("log_path", "logs/external_calls.jsonl")

    def process(self, request: Request) -> Response:
        """Process via external API."""

        if not self.api_key:
            return Response.error_response(
                error="External API key not configured (set EXTERNAL_API_KEY env var)",
                code="NO_API_KEY"
            )

        # Build prompt with attachments (v2.4)
        text = self._build_prompt_with_attachments(request)
        justification = request.metadata.get("justification", "No justification provided")

        # LOG BEFORE CALLING
        self._log_call(text, justification, "started")

        if self.provider == "anthropic":
            result = self._call_anthropic(text)
        elif self.provider == "openai":
            result = self._call_openai(text)
        else:
            return Response.error_response(
                error=f"Unknown provider: {self.provider}",
                code="UNKNOWN_PROVIDER"
            )

        # LOG AFTER CALLING
        self._log_call(text, justification, "completed", result.success)

        return result

    def _build_prompt_with_attachments(self, request: Request) -> str:
        """
        Build prompt including attachment content.

        SECURITY: Attachments are UNTRUSTED INERT DATA - never executed.

        Args:
            request: The request with potential attachments

        Returns:
            Formatted prompt string
        """
        text = request.text
        attachments = getattr(request, 'attachments', []) or []

        if not attachments:
            return text

        # Extract attachment context
        context_parts = [text, "\n\n--- ATTACHED FILES (User-provided content for analysis) ---"]

        for i, attachment in enumerate(attachments, 1):
            try:
                name = getattr(attachment, 'name', 'unknown')
                mime_type = getattr(attachment, 'type', 'unknown')
                size = getattr(attachment, 'size', 0)
                data = getattr(attachment, 'data', '')

                is_text = getattr(attachment, 'is_text', lambda: False)()
                is_code = getattr(attachment, 'is_code', lambda: False)()
                is_image = getattr(attachment, 'is_image', lambda: False)()

                if is_text or is_code:
                    try:
                        if data:
                            if data.startswith('data:'):
                                data = data.split(',', 1)[1] if ',' in data else data
                            decoded = base64.b64decode(data).decode('utf-8', errors='replace')
                            max_content = 8000
                            if len(decoded) > max_content:
                                decoded = decoded[:max_content] + f"\n... (truncated)"
                            context_parts.append(f"\n[File {i}: {name}]\nType: {mime_type}\nContent:\n```\n{decoded}\n```")
                        else:
                            context_parts.append(f"\n[File {i}: {name}]\nType: {mime_type}\n(No content)")
                    except Exception:
                        context_parts.append(f"\n[File {i}: {name}]\nType: {mime_type}\n(Could not decode)")
                elif is_image:
                    size_kb = size / 1024 if size else 0
                    context_parts.append(f"\n[File {i}: {name}]\nType: {mime_type} (Image)\nSize: {size_kb:.1f} KB")
                else:
                    size_kb = size / 1024 if size else 0
                    context_parts.append(f"\n[File {i}: {name}]\nType: {mime_type}\nSize: {size_kb:.1f} KB")
            except Exception as e:
                context_parts.append(f"\n[File {i}]: Error: {e}")

        context_parts.append("--- END ATTACHED FILES ---")
        return "\n".join(context_parts)

    def _call_anthropic(self, text: str) -> Response:
        """Call Claude API."""
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)

            response = client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": text}]
            )

            output_text = response.content[0].text

            return Response(
                success=True,
                text=output_text,
                output={
                    "content": output_text,
                    "model": self.model,
                    "provider": "anthropic",
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens
                    }
                },
                meta=ResponseMeta(used_external=True)
            )

        except ImportError:
            return Response.error_response(
                error="anthropic package not installed. Run: pip install anthropic",
                code="DEPENDENCY_MISSING"
            )
        except Exception as e:
            return Response.error_response(
                error=f"Anthropic API error: {str(e)}",
                code="ANTHROPIC_ERROR"
            )

    def _call_openai(self, text: str) -> Response:
        """Call OpenAI API."""
        try:
            import openai

            client = openai.OpenAI(api_key=self.api_key)

            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": text}]
            )

            output_text = response.choices[0].message.content

            return Response(
                success=True,
                text=output_text,
                output={
                    "content": output_text,
                    "model": self.model,
                    "provider": "openai",
                    "usage": {
                        "input_tokens": response.usage.prompt_tokens,
                        "output_tokens": response.usage.completion_tokens
                    }
                },
                meta=ResponseMeta(used_external=True)
            )

        except ImportError:
            return Response.error_response(
                error="openai package not installed. Run: pip install openai",
                code="DEPENDENCY_MISSING"
            )
        except Exception as e:
            return Response.error_response(
                error=f"OpenAI API error: {str(e)}",
                code="OPENAI_ERROR"
            )

    def _log_call(self, text: str, justification: str, status: str, success: bool = None):
        """Log external API call."""
        log_entry = {
            "timestamp": time.time(),
            "provider": self.provider,
            "model": self.model,
            "input_length": len(text),
            "justification": justification,
            "status": status
        }

        if success is not None:
            log_entry["success"] = success

        try:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except IOError:
            pass  # Don't fail on logging errors

    def is_available(self) -> bool:
        """Check if external API is configured."""
        return bool(self.api_key)
