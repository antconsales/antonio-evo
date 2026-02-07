"""
Mistral Handler - Local LLM via Ollama

ROLE: Reasoning, classification, text generation.
NOT ALLOWED: Tool execution, autonomous decisions.

Now supports:
- Configurable system prompt (for SOCIAL/LOGIC personas)
- Configurable temperature
"""

import json
import base64
import requests
from typing import Dict, Any, List

from .base import BaseHandler
from ..models.request import Request
from ..models.response import Response, ResponseMeta


class MistralHandler(BaseHandler):
    """
    Mistral 7B via Ollama.

    This is the main reasoning engine, but it CANNOT:
    - Execute tools
    - Make autonomous decisions
    - Call other handlers
    - Access external systems

    Supports different personas via configurable system prompts.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.model = config.get("model", "mistral")
        self.timeout = config.get("timeout", 60)

        # Persona support: configurable prompt and temperature
        self.system_prompt_path = config.get(
            "system_prompt_path",
            "prompts/mistral_system.txt"
        )
        self.temperature = config.get("temperature", 0.3)
        self.max_tokens = config.get("max_tokens", 1024)
        self.persona_name = config.get("persona_name", "default")

        self._system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        """Load system prompt from configured file."""
        # Force conversational prompt for SOCIAL persona
        if self.persona_name in ("SOCIAL", "default"):
            return "Sei Antonio, un assistente AI amichevole. Rispondi in modo naturale nella lingua dell'utente. NON analizzare - rispondi direttamente alla domanda come farebbe un amico."

        try:
            with open(self.system_prompt_path, encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return "You are a helpful assistant."

    def process(self, request: Request) -> Response:
        """Process text request via Mistral."""

        user_message = request.text
        task_type = request.task_type
        attachments = getattr(request, 'attachments', []) or []  # v2.4

        # DEBUG: Log attachment info
        if attachments:
            print(f"[MISTRAL DEBUG] Found {len(attachments)} attachments:")
            for i, att in enumerate(attachments):
                print(f"  [{i}] name={getattr(att, 'name', '?')}, type={getattr(att, 'type', '?')}, "
                      f"size={getattr(att, 'size', 0)}, has_data={bool(getattr(att, 'data', ''))}")

        if not user_message:
            return Response.error_response(
                error="No text provided",
                code="MISSING_TEXT"
            )

        # Build prompt based on task type with attachments (v2.4)
        prompt = self._build_prompt(task_type, user_message, attachments)

        # DEBUG: Log prompt length
        if attachments:
            print(f"[MISTRAL DEBUG] Prompt length: {len(prompt)} chars")

        try:
            # Use /api/chat for better conversation handling
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self._system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens
                    }
                },
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            output = result.get("message", {}).get("content", "")

            # Try to parse as JSON
            parsed_output = self._try_parse_json(output)

            return Response(
                success=True,
                output=parsed_output if parsed_output else output,
                text=output,
                meta=ResponseMeta()
            )

        except requests.Timeout:
            return Response.error_response(
                error="Ollama request timeout",
                code="TIMEOUT"
            )
        except requests.ConnectionError:
            return Response.error_response(
                error="Cannot connect to Ollama. Is it running?",
                code="CONNECTION_ERROR"
            )
        except requests.RequestException as e:
            return Response.error_response(
                error=str(e),
                code="OLLAMA_ERROR"
            )

    def _build_prompt(self, task_type: str, user_message: str, attachments: List = None) -> str:
        """Build task-specific prompt with optional attachments (v2.4)."""

        # Add attachment context if present
        attachment_context = ""
        if attachments:
            attachment_context = self._extract_attachment_context(attachments)

        if task_type == "classify":
            return f"""TASK: CLASSIFY

INPUT: {user_message}{attachment_context}

Respond with JSON containing: intent, domain, complexity, requires_external, confidence, reasoning."""

        elif task_type == "plan":
            return f"""TASK: PLAN

INPUT: {user_message}{attachment_context}

Decompose this into steps. Respond with JSON containing: steps (array), requires_external, reasoning."""

        elif task_type == "generate":
            return f"""TASK: GENERATE TEXT

INPUT: {user_message}{attachment_context}

Generate the requested content. Respond with JSON containing: content, tone, word_count."""

        elif task_type == "reason":
            return f"""TASK: REASON

INPUT: {user_message}{attachment_context}

Analyze and respond with JSON containing: analysis, conclusion, confidence, caveats."""

        else:  # chat (default) - natural conversation
            return f"{user_message}{attachment_context}"

    def _extract_attachment_context(self, attachments: List) -> str:
        """
        Extract content from attachments for LLM context.

        SECURITY: Attachments are UNTRUSTED INERT DATA - never executed.

        Args:
            attachments: List of Attachment objects

        Returns:
            Formatted string with attachment content
        """
        if not attachments:
            return ""

        context_parts = ["\n\n--- ATTACHED FILES ---"]

        for i, att in enumerate(attachments, 1):
            try:
                name = getattr(att, 'name', 'unknown')
                mime_type = getattr(att, 'type', 'unknown')
                size = getattr(att, 'size', 0)
                data = getattr(att, 'data', '')

                is_text = getattr(att, 'is_text', lambda: False)()
                is_code = getattr(att, 'is_code', lambda: False)()
                is_image = getattr(att, 'is_image', lambda: False)()

                if is_text or is_code:
                    try:
                        if data:
                            if data.startswith('data:'):
                                data = data.split(',', 1)[1] if ',' in data else data
                            decoded = base64.b64decode(data).decode('utf-8', errors='replace')
                            max_content = 6000
                            if len(decoded) > max_content:
                                decoded = decoded[:max_content] + "\n... (truncated)"
                            context_parts.append(f"\n[{name}]\n```\n{decoded}\n```")
                        else:
                            context_parts.append(f"\n[{name}] (empty)")
                    except Exception:
                        context_parts.append(f"\n[{name}] (could not decode)")
                elif is_image:
                    context_parts.append(f"\n[{name}] (Image: {mime_type}, {size/1024:.1f}KB)")
                else:
                    context_parts.append(f"\n[{name}] ({mime_type})")
            except Exception:
                continue

        context_parts.append("--- END FILES ---")
        return "\n".join(context_parts)

    def _try_parse_json(self, text: str) -> Any:
        """Try to extract and parse JSON from text."""
        try:
            # Try direct parse
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from text
        try:
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(text[json_start:json_end])
        except json.JSONDecodeError:
            pass

        return None

    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
