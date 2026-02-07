"""
Local LLM Handler - Offline text processing via Ollama.

This handler provides local LLM inference using Ollama (Mistral 7B default).
It is designed for offline operation and runs within the sandbox.

CAPABILITIES:
- Text generation and completion
- Question answering
- Text analysis and summarization
- Structured JSON output when requested

RESTRICTIONS (enforced by system prompt):
- No tool calling or function execution
- No hallucinated capabilities (e.g., web search, file access)
- No routing decisions
- No filesystem access
- No network access beyond Ollama

DESIGN PRINCIPLES:
- Never raises uncaught exceptions
- Respects configured timeouts
- Deterministic output (low temperature default)
- Stateless operation
"""

import json
import base64
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, List

import requests

from .base import BaseHandler
from ..models.request import Request
from ..models.response import Response, ResponseMeta


class OutputFormat(Enum):
    """Supported output formats."""
    TEXT = "text"
    JSON = "json"


class LLMLocalErrorCode(Enum):
    """Error codes for local LLM handler."""
    MISSING_TEXT = "LLM_MISSING_TEXT"
    CONNECTION_ERROR = "LLM_CONNECTION_ERROR"
    TIMEOUT = "LLM_TIMEOUT"
    OLLAMA_ERROR = "LLM_OLLAMA_ERROR"
    MALFORMED_RESPONSE = "LLM_MALFORMED_RESPONSE"
    JSON_PARSE_ERROR = "LLM_JSON_PARSE_ERROR"


@dataclass
class LLMLocalConfig:
    """
    Configuration for the local LLM handler.

    Attributes:
        base_url: Ollama server URL (default: http://localhost:11434)
        model: Model name to use (default: mistral)
        timeout_seconds: Request timeout in seconds (default: 60)
        temperature: Sampling temperature, lower = more deterministic (default: 0.1)
        max_tokens: Maximum tokens to generate (default: 1024)
        output_format: Default output format (default: TEXT)
    """
    base_url: str = "http://localhost:11434"
    model: str = "mistral"
    timeout_seconds: int = 60
    temperature: float = 0.1
    max_tokens: int = 1024
    output_format: OutputFormat = OutputFormat.TEXT

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "LLMLocalConfig":
        """Create config from dictionary."""
        output_format_str = config.get("output_format", "text")
        try:
            output_format = OutputFormat(output_format_str)
        except ValueError:
            output_format = OutputFormat.TEXT

        return cls(
            base_url=config.get("base_url", cls.base_url),
            model=config.get("model", cls.model),
            timeout_seconds=config.get("timeout_seconds", config.get("timeout", cls.timeout_seconds)),
            temperature=config.get("temperature", cls.temperature),
            max_tokens=config.get("max_tokens", cls.max_tokens),
            output_format=output_format,
        )


# System prompt that enforces safety rules
SYSTEM_PROMPT = """You are a helpful, accurate assistant running locally.

IMPORTANT RULES - YOU MUST FOLLOW THESE:
1. You CANNOT call tools or functions - you have no tools available.
2. You CANNOT browse the web, search the internet, or access URLs.
3. You CANNOT read, write, or access any files on the filesystem.
4. You CANNOT execute code or commands.
5. You CANNOT access external APIs or services.
6. You MUST NOT claim capabilities you don't have.
7. You MUST admit when you don't know something.

YOUR CAPABILITIES:
- Answer questions based on your training knowledge
- Generate text, summaries, and explanations
- Analyze and discuss text provided to you
- Provide structured JSON output when requested

RESPONSE STYLE:
- Be concise and factual
- Avoid speculation and hallucination
- If uncertain, say so clearly
- When asked for JSON, respond ONLY with valid JSON (no markdown, no explanation)"""


class LLMLocalHandler(BaseHandler):
    """
    Local LLM handler using Ollama.

    Provides offline text generation and processing using a local
    Ollama instance running Mistral 7B (or other configured model).

    This handler is designed to:
    - Run within the sandbox with resource limits
    - Never raise uncaught exceptions
    - Return structured Response objects
    - Support both plain text and JSON output

    Usage:
        handler = LLMLocalHandler({"base_url": "http://localhost:11434"})
        response = handler.process(request)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the local LLM handler.

        Args:
            config: Handler configuration dictionary with keys:
                - base_url: Ollama server URL
                - model: Model name (default: mistral)
                - timeout_seconds: Request timeout
                - temperature: Sampling temperature
                - max_tokens: Maximum output tokens
                - output_format: "text" or "json"
        """
        super().__init__(config)
        self._config = LLMLocalConfig.from_dict(config)

    def _validate_config(self):
        """Validate handler configuration."""
        # Config validation is handled by LLMLocalConfig.from_dict
        # which applies safe defaults for invalid values
        pass

    def process(self, request: Request) -> Response:
        """
        Process a text request using the local LLM.

        Args:
            request: Request object with text to process

        Returns:
            Response object with:
            - success=True and output/text on success
            - success=False and error/error_code on failure

        Note:
            This method NEVER raises exceptions. All errors are
            returned as Response objects with appropriate error codes.
        """
        # Validate input
        if not request.text or not request.text.strip():
            return Response.error_response(
                error="No text provided for LLM processing",
                code=LLMLocalErrorCode.MISSING_TEXT.value,
            )

        # Determine output format from request metadata
        output_format = self._get_output_format(request)

        # Get attachments if present (v2.4)
        attachments = getattr(request, 'attachments', []) or []

        # Build the prompt with attachment context
        prompt = self._build_prompt(request.text, output_format, attachments)

        # Make the Ollama request
        return self._call_ollama(prompt, output_format)

    def _get_output_format(self, request: Request) -> OutputFormat:
        """
        Determine output format from request.

        Args:
            request: The incoming request

        Returns:
            OutputFormat enum value
        """
        # Check request metadata for format preference
        metadata = request.metadata or {}
        format_str = metadata.get("output_format", "")

        if format_str.lower() == "json":
            return OutputFormat.JSON

        # Check if text explicitly asks for JSON
        text_lower = request.text.lower() if request.text else ""
        if "respond in json" in text_lower or "return json" in text_lower:
            return OutputFormat.JSON

        return self._config.output_format

    def _build_prompt(self, user_text: str, output_format: OutputFormat, attachments: List = None) -> str:
        """
        Build the prompt for the LLM.

        Args:
            user_text: User's input text
            output_format: Desired output format
            attachments: Optional list of Attachment objects

        Returns:
            Formatted prompt string
        """
        # Start with user text
        prompt_parts = [user_text]

        # Add attachment context if present (v2.4)
        if attachments:
            attachment_context = self._extract_attachment_context(attachments)
            if attachment_context:
                prompt_parts.append("\n\n--- ATTACHED FILES (User-provided content for analysis) ---")
                prompt_parts.append(attachment_context)
                prompt_parts.append("--- END ATTACHED FILES ---")

        prompt = "\n".join(prompt_parts)

        if output_format == OutputFormat.JSON:
            return f"{prompt}\n\nRespond with valid JSON only. No explanation, no markdown."
        return prompt

    def _extract_attachment_context(self, attachments: List) -> str:
        """
        Extract readable content from attachments for LLM context.

        SECURITY: Attachments are treated as UNTRUSTED INERT DATA.
        - Text content is extracted and included verbatim
        - Images are described (filename, type, size) but not processed here
        - Binary files are only described by metadata
        - Content is NEVER executed, only analyzed as text

        Args:
            attachments: List of Attachment objects

        Returns:
            Formatted string with attachment content
        """
        context_parts = []

        for i, attachment in enumerate(attachments, 1):
            try:
                name = getattr(attachment, 'name', 'unknown')
                mime_type = getattr(attachment, 'type', 'unknown')
                size = getattr(attachment, 'size', 0)
                data = getattr(attachment, 'data', '')

                # Check attachment type methods
                is_text = getattr(attachment, 'is_text', lambda: False)()
                is_code = getattr(attachment, 'is_code', lambda: False)()
                is_image = getattr(attachment, 'is_image', lambda: False)()

                if is_text or is_code:
                    # Extract text content from base64
                    try:
                        if data:
                            # Handle data URL format (data:type;base64,content)
                            if data.startswith('data:'):
                                data = data.split(',', 1)[1] if ',' in data else data
                            decoded = base64.b64decode(data).decode('utf-8', errors='replace')
                            # Limit content length to avoid token overflow
                            max_content = 8000  # Characters
                            if len(decoded) > max_content:
                                decoded = decoded[:max_content] + f"\n... (truncated, {len(decoded) - max_content} more characters)"
                            context_parts.append(f"\n[File {i}: {name}]\nType: {mime_type}\nContent:\n```\n{decoded}\n```")
                        else:
                            context_parts.append(f"\n[File {i}: {name}]\nType: {mime_type}\n(No content available)")
                    except Exception as e:
                        context_parts.append(f"\n[File {i}: {name}]\nType: {mime_type}\n(Could not decode content: {e})")

                elif is_image:
                    # For images, include metadata only (vision analysis would need separate handler)
                    size_kb = size / 1024 if size else 0
                    context_parts.append(f"\n[File {i}: {name}]\nType: {mime_type} (Image)\nSize: {size_kb:.1f} KB\n(Image attached - describe based on user's question)")

                else:
                    # Binary or unknown file - include metadata only
                    size_kb = size / 1024 if size else 0
                    context_parts.append(f"\n[File {i}: {name}]\nType: {mime_type}\nSize: {size_kb:.1f} KB\n(Binary file - cannot read content)")

            except Exception as e:
                context_parts.append(f"\n[File {i}]: Error processing attachment: {e}")

        return "\n".join(context_parts) if context_parts else ""

    def _call_ollama(self, prompt: str, output_format: OutputFormat) -> Response:
        """
        Make a request to the Ollama API.

        Args:
            prompt: The formatted prompt
            output_format: Expected output format

        Returns:
            Response object with result or error

        Note:
            This method catches ALL exceptions to ensure no uncaught
            exceptions escape the handler.
        """
        try:
            response = requests.post(
                f"{self._config.base_url}/api/generate",
                json={
                    "model": self._config.model,
                    "prompt": prompt,
                    "system": SYSTEM_PROMPT,
                    "stream": False,
                    "options": {
                        "temperature": self._config.temperature,
                        "num_predict": self._config.max_tokens,
                    },
                },
                timeout=self._config.timeout_seconds,
            )
            response.raise_for_status()

            return self._parse_ollama_response(response, output_format)

        except requests.Timeout:
            return Response.error_response(
                error=f"Ollama request timed out after {self._config.timeout_seconds}s",
                code=LLMLocalErrorCode.TIMEOUT.value,
            )

        except requests.ConnectionError:
            return Response.error_response(
                error=f"Cannot connect to Ollama at {self._config.base_url}. Is it running?",
                code=LLMLocalErrorCode.CONNECTION_ERROR.value,
            )

        except requests.HTTPError as e:
            return Response.error_response(
                error=f"Ollama HTTP error: {e.response.status_code}",
                code=LLMLocalErrorCode.OLLAMA_ERROR.value,
            )

        except requests.RequestException as e:
            return Response.error_response(
                error=f"Ollama request failed: {str(e)}",
                code=LLMLocalErrorCode.OLLAMA_ERROR.value,
            )

        except Exception as e:
            # Catch-all to ensure no exceptions escape
            return Response.error_response(
                error=f"Unexpected error in LLM handler: {str(e)}",
                code=LLMLocalErrorCode.OLLAMA_ERROR.value,
            )

    def _parse_ollama_response(
        self,
        http_response: requests.Response,
        output_format: OutputFormat,
    ) -> Response:
        """
        Parse the Ollama API response.

        Args:
            http_response: The HTTP response from Ollama
            output_format: Expected output format

        Returns:
            Response object with parsed content
        """
        try:
            result = http_response.json()
        except json.JSONDecodeError:
            return Response.error_response(
                error="Ollama returned invalid JSON response",
                code=LLMLocalErrorCode.MALFORMED_RESPONSE.value,
            )

        # Extract the response text
        response_text = result.get("response", "")

        if not response_text:
            return Response.error_response(
                error="Ollama returned empty response",
                code=LLMLocalErrorCode.MALFORMED_RESPONSE.value,
            )

        # Handle JSON output format
        if output_format == OutputFormat.JSON:
            return self._parse_json_output(response_text)

        # Plain text output
        return Response(
            success=True,
            output=response_text,
            text=response_text,
            meta=ResponseMeta(handler=self.name),
        )

    def _parse_json_output(self, response_text: str) -> Response:
        """
        Parse JSON output from LLM response.

        Args:
            response_text: Raw text from LLM

        Returns:
            Response with parsed JSON output or error
        """
        # Try direct JSON parse
        try:
            parsed = json.loads(response_text.strip())
            return Response(
                success=True,
                output=parsed,
                text=response_text,
                meta=ResponseMeta(handler=self.name),
            )
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from text (LLM sometimes adds extra text)
        extracted = self._extract_json(response_text)
        if extracted is not None:
            return Response(
                success=True,
                output=extracted,
                text=response_text,
                meta=ResponseMeta(handler=self.name),
            )

        # Failed to parse as JSON - return as text with warning
        return Response(
            success=True,
            output=response_text,
            text=response_text,
            warnings=["Requested JSON format but LLM returned plain text"],
            meta=ResponseMeta(handler=self.name),
        )

    def _extract_json(self, text: str) -> Optional[Any]:
        """
        Extract JSON object or array from text.

        Handles cases where LLM wraps JSON in markdown code blocks
        or adds explanatory text around the JSON.

        Args:
            text: Text potentially containing JSON

        Returns:
            Parsed JSON if found, None otherwise
        """
        # Remove markdown code blocks if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last lines (code block markers)
            if len(lines) >= 2:
                lines = lines[1:]  # Remove opening ```json or ```
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]  # Remove closing ```
                cleaned = "\n".join(lines).strip()

        # Try to find JSON object
        try:
            json_start = cleaned.find("{")
            json_end = cleaned.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(cleaned[json_start:json_end])
        except json.JSONDecodeError:
            pass

        # Try to find JSON array
        try:
            json_start = cleaned.find("[")
            json_end = cleaned.rfind("]") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(cleaned[json_start:json_end])
        except json.JSONDecodeError:
            pass

        return None

    def is_available(self) -> bool:
        """
        Check if Ollama is running and accessible.

        Returns:
            True if Ollama is available, False otherwise

        Note:
            This method never raises exceptions.
        """
        try:
            response = requests.get(
                f"{self._config.base_url}/api/tags",
                timeout=5,
            )
            return response.status_code == 200
        except Exception:
            return False

    def check_model_available(self) -> bool:
        """
        Check if the configured model is available in Ollama.

        Returns:
            True if model is available, False otherwise

        Note:
            This method never raises exceptions.
        """
        try:
            response = requests.get(
                f"{self._config.base_url}/api/tags",
                timeout=5,
            )
            if response.status_code != 200:
                return False

            data = response.json()
            models = data.get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]
            return self._config.model in model_names

        except Exception:
            return False
