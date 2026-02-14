"""
Mistral Handler - Local LLM via Ollama

ROLE: Reasoning, classification, text generation, and agentic tool use.

Supports:
- Configurable system prompt (for SOCIAL/LOGIC personas)
- Configurable temperature
- ReAct loop with native Ollama tool calling (v5.0)
"""

import json
import logging
import base64
import requests
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 8

from .base import BaseHandler
from ..models.request import Request
from ..models.response import Response, ResponseMeta


class MistralHandler(BaseHandler):
    """
    Local LLM via Ollama (Qwen3/Mistral).

    Main reasoning engine with agentic tool-use (v5.0).
    Supports ReAct loop: LLM -> tool_calls -> execute -> re-prompt.
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

        # Qwen3 thinking mode: disabled on CPU for performance (~6x faster)
        # Enable only if explicitly configured (e.g. with GPU)
        self.enable_thinking = config.get("enable_thinking", False)

        self._system_prompt = self._load_system_prompt()

        # Document parser (injected by Router, optional)
        self._document_parser = None
        # Vision service (injected by Router, optional)
        self._vision_service = None
        # Tool system (v5.0 - injected by Orchestrator)
        self._tool_registry = None
        self._tool_executor = None

    def set_document_parser(self, parser):
        """Inject document parser service for image/PDF OCR."""
        self._document_parser = parser

    def set_vision_service(self, service):
        """Inject vision service for image understanding (SmolVLM2)."""
        self._vision_service = service

    def set_tool_registry(self, registry):
        """Inject tool registry for function calling (v5.0)."""
        self._tool_registry = registry

    def set_tool_executor(self, executor):
        """Inject tool executor for function calling (v5.0)."""
        self._tool_executor = executor

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

    def _get_system_prompt_with_tools(self) -> str:
        """Augment system prompt with tool usage instructions if tools available."""
        base = self._system_prompt
        if not self._tool_registry or len(self._tool_registry) == 0:
            return base

        import os
        cwd = os.getcwd()

        tools_instruction = (
            "\n\nYou have access to tools you can call to help the user. "
            "When a question requires current information, files, code execution, or image analysis, "
            "use the appropriate tool. You can use multiple tools in sequence. "
            "After getting tool results, synthesize them into a clear response. "
            "Do NOT mention that you're using tools unless the user asks."
            f"\n\nIMPORTANT: The current working directory is: {cwd}\n"
            "For file operations, use relative paths (e.g. 'README.md', 'src/main.py') or "
            "absolute paths starting with the CWD. Use 'list_directory' with path '.' to see "
            "available files. For code execution, the code runs in the CWD."
        )
        return base + tools_instruction

    def process(self, request: Request) -> Response:
        """Process text request via LLM with optional ReAct tool-use loop (v5.0)."""

        user_message = request.text
        task_type = request.task_type
        attachments = getattr(request, 'attachments', []) or []
        metadata = getattr(request, 'metadata', {}) or {}

        if not user_message:
            return Response.error_response(error="No text provided", code="MISSING_TEXT")

        # Build initial prompt
        prompt = self._build_prompt(task_type, user_message, attachments, metadata)

        # Determine if tools should be used for this task type
        use_tools = (
            self._tool_registry is not None
            and self._tool_executor is not None
            and len(self._tool_registry) > 0
            and task_type in ("chat", "reason", "plan", None, "")
        )

        # Get tool callback from request metadata (injected by pipeline)
        tool_callback = metadata.get("_tool_callback")

        # Build system prompt (with tool instructions if available)
        system_prompt = self._get_system_prompt_with_tools() if use_tools else self._system_prompt

        # Build initial messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        # Thinking mode
        use_thinking = self.enable_thinking
        if request.metadata and 'think' in request.metadata:
            use_thinking = request.metadata['think']

        # Tool definitions for Ollama
        tool_definitions = self._tool_registry.get_definitions() if use_tools else []

        # Track tools used for response metadata
        tools_used = []

        try:
            # === ReAct Loop ===
            for iteration in range(MAX_TOOL_ITERATIONS):
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                }

                if not use_thinking:
                    payload["think"] = False

                # Include tool definitions
                if use_tools and tool_definitions:
                    payload["tools"] = tool_definitions

                logger.debug(f"ReAct iteration {iteration + 1}, messages={len(messages)}, tools={'yes' if tool_definitions else 'no'}")

                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()

                result = response.json()
                message = result.get("message", {})
                tool_calls = message.get("tool_calls", [])

                if not tool_calls:
                    # No tool calls - LLM is done, return final text
                    output = message.get("content", "")
                    parsed_output = self._try_parse_json(output)
                    meta = ResponseMeta()
                    if tools_used:
                        meta.tools_used = tools_used
                    return Response(
                        success=True,
                        output=parsed_output if parsed_output else output,
                        text=output,
                        meta=meta,
                    )

                # Tool calls detected - execute each
                messages.append(message)
                logger.info(f"ReAct iteration {iteration + 1}: {len(tool_calls)} tool call(s)")

                for tool_call in tool_calls:
                    func = tool_call.get("function", {})
                    tool_name = func.get("name", "")
                    arguments = func.get("arguments", {})

                    # Parse arguments if string
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except (json.JSONDecodeError, TypeError):
                            arguments = {}

                    # Execute tool
                    tool_result = self._tool_executor.execute(
                        tool_name, arguments, callback=tool_callback
                    )

                    # Track for response metadata
                    tools_used.append({
                        "tool": tool_name,
                        "arguments": {k: str(v)[:100] for k, v in arguments.items()},
                        "success": tool_result.success,
                        "elapsed_ms": tool_result.elapsed_ms,
                    })

                    # Append tool result for next LLM iteration
                    messages.append({
                        "role": "tool",
                        "content": tool_result.output,
                    })

            # Max iterations reached
            last_content = "I used the maximum number of tool calls. Here's what I found so far."
            for msg in reversed(messages):
                if msg.get("role") == "tool" and msg.get("content"):
                    last_content += "\n\n" + msg["content"]
                    break
            meta = ResponseMeta()
            if tools_used:
                meta.tools_used = tools_used
            return Response(
                success=True,
                output=last_content,
                text=last_content,
                meta=meta,
            )

        except requests.Timeout:
            return Response.error_response(error="Ollama request timeout", code="TIMEOUT")
        except requests.ConnectionError:
            return Response.error_response(error="Cannot connect to Ollama. Is it running?", code="CONNECTION_ERROR")
        except requests.RequestException as e:
            return Response.error_response(error=str(e), code="OLLAMA_ERROR")

    def _build_prompt(self, task_type: str, user_message: str, attachments: List = None, metadata: Dict = None) -> str:
        """Build task-specific prompt with optional attachments (v2.4) and web search."""

        # Add attachment context if present
        attachment_context = ""
        if attachments:
            attachment_context = self._extract_attachment_context(attachments)

        # Add web search context if available (from pipeline preprocessing)
        web_context = ""
        if metadata and '_web_search' in metadata:
            web_context = metadata['_web_search']

        extra_context = attachment_context + web_context

        if task_type == "classify":
            return f"""TASK: CLASSIFY

INPUT: {user_message}{extra_context}

Respond with JSON containing: intent, domain, complexity, requires_external, confidence, reasoning."""

        elif task_type == "plan":
            return f"""TASK: PLAN

INPUT: {user_message}{extra_context}

Decompose this into steps. Respond with JSON containing: steps (array), requires_external, reasoning."""

        elif task_type == "generate":
            return f"""TASK: GENERATE TEXT

INPUT: {user_message}{extra_context}

Generate the requested content. Respond with JSON containing: content, tone, word_count."""

        elif task_type == "reason":
            return f"""TASK: REASON

INPUT: {user_message}{extra_context}

Analyze and respond with JSON containing: analysis, conclusion, confidence, caveats."""

        else:  # chat (default) - natural conversation
            return f"{user_message}{extra_context}"

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

                is_pdf = getattr(att, 'is_pdf', lambda: False)()

                # Check for pre-analyzed description (set by pipeline preprocessing)
                pre_description = getattr(att, 'description', None)

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
                elif is_image or is_pdf:
                    # Use pre-analyzed description if available (from pipeline)
                    if pre_description:
                        context_parts.append(f"\n[{name}] {pre_description}")
                        continue

                    # Fallback: try VisionService in-process (may fail in sandbox)
                    if is_image and self._vision_service:
                        try:
                            vision_result = self._vision_service.analyze_attachment(att)
                            if vision_result.success and vision_result.description:
                                context_parts.append(
                                    f"\n[{name}] (Image analyzed by {vision_result.model}, {vision_result.elapsed_ms}ms)\n"
                                    f"Description: {vision_result.description}"
                                )
                                continue
                        except Exception as e:
                            print(f"[MISTRAL] Vision service error: {e}")

                    # Fallback: try document parsing via dots.ocr
                    parse_result = None
                    if self._document_parser:
                        try:
                            parse_result = self._document_parser.parse_attachment(att)
                        except Exception as e:
                            print(f"[MISTRAL] Document parser error: {e}")

                    if parse_result and parse_result.success and parse_result.text:
                        label = "PDF content" if is_pdf else "Image text"
                        truncated = parse_result.text[:8000]
                        if len(parse_result.text) > 8000:
                            truncated += "\n... (truncated)"
                        context_parts.append(
                            f"\n[{name}] ({label} extracted via OCR, {parse_result.elapsed_ms}ms)\n```\n{truncated}\n```"
                        )
                    elif is_image:
                        context_parts.append(f"\n[{name}] (Image: {mime_type}, {size/1024:.1f}KB - no vision model available)")
                    else:
                        context_parts.append(f"\n[{name}] (PDF: {size/1024:.1f}KB, OCR not available)")
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
