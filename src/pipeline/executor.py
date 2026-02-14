"""
Pipeline Executor - The deterministic 8-step processing pipeline.

Extracted from Orchestrator to follow Single Responsibility Principle.

Flow (deterministic, no exceptions):
1. Normalize input -> Request
2. Retrieve memory context -> MemoryContext
3. Get emotional context (v2.1)
4. Classify intent (memory-informed) -> Classification
5. Apply policy -> PolicyDecision
6. Route to handler -> Response
7. Create neuron + post-processing (emotional, proactive, digital twin)
8. Log to audit trail
"""

import time
import logging
from typing import Dict, Any, Union, Optional

logger = logging.getLogger(__name__)


class PipelineExecutor:
    """
    Executes the deterministic request processing pipeline.

    All components are injected via constructor - no hidden dependencies.
    """

    def __init__(
        self,
        normalizer,
        classifier,
        policy,
        router,
        response_builder,
        audit,
        session_manager,
        memory_enabled: bool = False,
        memory_retriever=None,
        neuron_creator=None,
        emotional_memory=None,
        proactive_service=None,
        digital_twin=None,
        webhook_service=None,
        web_search_service=None,
        rag_client=None,
        rag_config=None,
    ):
        """
        Initialize pipeline with all required components.

        All parameters are injected - no internal instantiation.
        """
        self.normalizer = normalizer
        self.classifier = classifier
        self.policy = policy
        self.router = router
        self.response_builder = response_builder
        self.audit = audit
        self.session_manager = session_manager

        # Optional memory components
        self.memory_enabled = memory_enabled
        self.memory_retriever = memory_retriever
        self.neuron_creator = neuron_creator
        self.emotional_memory = emotional_memory
        self.proactive_service = proactive_service
        self.digital_twin = digital_twin

        # Webhook service (n8n integration)
        self.webhook_service = webhook_service

        # Web search service (Tavily)
        self.web_search_service = web_search_service

        # RAG client + config (v7.0 knowledge base)
        self.rag_client = rag_client
        self.rag_config = rag_config or {}

        # Hook registry (v6.0 plugin system)
        self.hook_registry = None

    def execute(self, raw_input: Union[str, Dict[str, Any]], tool_callback=None, chunk_callback=None) -> Dict[str, Any]:
        """
        Process a single request through the 8-step pipeline.

        SYNCHRONOUS. One request in, one response out.
        No async, no callbacks, no side effects beyond audit log and memory.

        Args:
            raw_input: User input (string or dict with text, attachments, etc.)
            tool_callback: Optional callback for tool action events (v5.0)
            chunk_callback: Optional callback for streaming text chunks (v6.0)
        """
        start_time = time.time()
        memory_operation = None

        # === STEP 1: Normalize ===
        norm_result = self.normalizer.normalize(raw_input)

        if not norm_result.success:
            return {
                "success": False,
                "error": "; ".join([e.message for e in norm_result.errors]),
                "error_code": norm_result.error_code,
                "elapsed_ms": int((time.time() - start_time) * 1000),
            }

        request = norm_result.request

        # Attach session ID
        request.session_id = self.session_manager.current_session_id

        # === STEP 2: Memory Retrieval ===
        if self.memory_enabled and self.memory_retriever:
            try:
                memory_context = self.memory_retriever.retrieve(
                    query=request.text,
                    session_id=request.session_id,
                )
                request.memory_context = memory_context
            except Exception:
                request.memory_context = None

        # === STEP 2.1: Emotional Context (v2.1) ===
        emotional_context = None
        if self.memory_enabled and self.emotional_memory:
            try:
                emotional_context = self.emotional_memory.get_emotional_context(
                    message=request.text,
                    session_id=request.session_id,
                )
                request.emotional_context = emotional_context
            except Exception:
                request.emotional_context = None

        # === STEP 2.2: Knowledge Context (v7.0 RAG auto-enrichment) ===
        if (self.rag_client and self.rag_client.is_available()
                and self.rag_config.get("auto_enrich", False)):
            try:
                results = self.rag_client.search(
                    query=request.text,
                    limit=self.rag_config.get("top_k", 3),
                )
                score_threshold = self.rag_config.get("score_threshold", 0.3)
                filtered = [r for r in results if r.score >= score_threshold]
                if filtered:
                    max_chars = self.rag_config.get("max_context_chars", 2000)
                    parts = []
                    total = 0
                    for r in filtered:
                        chunk = f"[{r.source}] {r.text}"
                        if total + len(chunk) > max_chars:
                            break
                        parts.append(chunk)
                        total += len(chunk)
                    request.knowledge_context = "\n---\n".join(parts)
                    logger.debug(f"RAG enrichment: {len(filtered)} docs, {total} chars")
            except Exception:
                request.knowledge_context = None

        # === STEP 2.5: Preprocess Image Attachments ===
        # Analyze images BEFORE routing to avoid sandbox subprocess issues
        # (VisionService HTTP calls fail inside Windows multiprocessing.Process)
        if request.attachments:
            self._preprocess_attachments(request)

        # === STEP 2.6: Inject tool callback (v5.0) ===
        # The ReAct loop in MistralHandler uses this to emit WebSocket events
        if tool_callback:
            request.metadata["_tool_callback"] = tool_callback

        # === STEP 2.7: Inject chunk callback (v6.0) ===
        # Streaming tokens from MistralHandler to WebSocket client
        if chunk_callback:
            request.metadata["_chunk_callback"] = chunk_callback

        # === STEP 2.7.1: Inject knowledge context into metadata (v7.0) ===
        if request.knowledge_context:
            request.metadata["_knowledge_context"] = request.knowledge_context

        # === STEP 2.8: Plugin pre_process hook (v6.0) ===
        if self.hook_registry:
            self.hook_registry.emit("pre_process", {"text": request.text, "session_id": request.session_id})

        # === STEP 3: Classify (memory-informed) ===
        classification = self.classifier.classify(request)

        # Enhance classification with memory info
        if request.memory_context and request.memory_context.has_relevant_memory:
            classification.memory_informed = True
            classification.memory_confidence = request.memory_context.avg_confidence

        # === STEP 4: Policy ===
        decision = self.policy.decide(request, classification)

        # === STEP 5: Route ===
        result = self.router.route(request, decision)

        # === STEP 6: Build response ===
        elapsed_ms = int((time.time() - start_time) * 1000)
        response = self.response_builder.build(
            result=result,
            decision=decision,
            classification=classification,
            elapsed_ms=elapsed_ms
        )

        # === STEP 6.1: Plugin post_process hook (v6.0) ===
        if self.hook_registry:
            self.hook_registry.emit("post_process", {
                "success": response.get("success", False) if isinstance(response, dict) else True,
                "handler": decision.handler.value if decision else None,
                "elapsed_ms": elapsed_ms,
            })

        # === STEP 7: Create Neuron (if successful) ===
        neuron = None
        if self.memory_enabled and self.neuron_creator:
            try:
                neuron = self.neuron_creator.create_neuron(
                    request=request,
                    response=response,
                    decision=decision,
                    classification=classification,
                )

                memory_operation = self.neuron_creator.create_memory_operation(
                    request=request,
                    neuron=neuron,
                )

                if neuron:
                    response["_meta"] = response.get("_meta", {})
                    response["_meta"]["neuron_stored"] = True
                    response["_meta"]["neuron_id"] = neuron.id
                    response["_meta"]["neuron_confidence"] = neuron.confidence

            except Exception:
                pass

        # === STEP 7.1: Add Emotional Context to Response (v2.1) ===
        if emotional_context:
            response["_meta"] = response.get("_meta", {})
            response["_meta"]["emotional"] = {
                "user_state": emotional_context.current_state.value,
                "confidence": round(emotional_context.current_confidence, 2),
                "tone_recommendation": emotional_context.tone_recommendation.value,
                "trend": emotional_context.emotional_trend,
                "notes": emotional_context.adaptation_notes[:2] if emotional_context.adaptation_notes else [],
            }

        # === STEP 7.2: Log Interaction for Proactive Analysis (v2.2) ===
        if self.proactive_service:
            try:
                self.proactive_service.log_interaction(
                    message=request.text,
                    session_id=request.session_id,
                    emotional_state=emotional_context.current_state.value if emotional_context else None,
                    response_success=response.get("success", False),
                )
            except Exception:
                pass

        # === STEP 7.3: Learn User Style for Digital Twin (v3.0) ===
        if self.digital_twin:
            try:
                self.digital_twin.learn_from_message(request.text)
            except Exception:
                pass

        # === STEP 8: Audit log (with memory operation) ===
        self.audit.log(
            request=request,
            classification=classification,
            decision=decision,
            response=response,
            elapsed_ms=elapsed_ms,
            memory_operation=memory_operation,
        )

        # === STEP 9: Webhook triggers (non-blocking) ===
        if self.webhook_service:
            try:
                event_data = {
                    "request_text": request.text[:500] if request.text else "",
                    "response_text": response.text[:500] if hasattr(response, 'text') and response.text else "",
                    "handler": decision.handler.value if decision else None,
                    "success": response.success if hasattr(response, 'success') else True,
                    "elapsed_ms": elapsed_ms,
                    "session_id": self.session_manager.current_session_id if self.session_manager else None,
                }
                self.webhook_service.trigger_event("post_response", event_data)

                # Trigger on_error for failed responses
                if hasattr(response, 'success') and not response.success:
                    self.webhook_service.trigger_event("on_error", event_data)
            except Exception:
                pass  # Webhooks must never break the pipeline

        return response

    def _preprocess_attachments(self, request) -> None:
        """
        Pre-analyze image attachments using VisionService.

        Runs in the main process (not sandbox) so HTTP calls to Ollama work
        reliably. Stores descriptions on Attachment.description for the handler.
        """
        vision_service = getattr(self.router, 'vision_service', None)
        document_parser = getattr(self.router, 'document_parser', None)

        for att in request.attachments:
            try:
                is_image = att.is_image() if hasattr(att, 'is_image') else False
                is_pdf = att.is_pdf() if hasattr(att, 'is_pdf') else False

                if is_image and vision_service:
                    logger.info(f"[Pipeline] Analyzing image: {att.name} via VisionService")
                    result = vision_service.analyze_attachment(att)
                    if result.success and result.description:
                        att.description = (
                            f"(Image analyzed by {result.model}, {result.elapsed_ms}ms)\n"
                            f"Description: {result.description}"
                        )
                        logger.info(f"[Pipeline] Image analyzed: {att.name} ({result.elapsed_ms}ms)")
                        continue
                    else:
                        logger.warning(f"[Pipeline] Vision failed for {att.name}: {result.error}")

                # Fallback: try document parser (dots.ocr) for images and PDFs
                if (is_image or is_pdf) and document_parser:
                    try:
                        parse_result = document_parser.parse_attachment(att)
                        if parse_result and parse_result.success and parse_result.text:
                            label = "PDF content" if is_pdf else "Image text"
                            truncated = parse_result.text[:8000]
                            if len(parse_result.text) > 8000:
                                truncated += "\n... (truncated)"
                            att.description = f"({label} extracted via OCR, {parse_result.elapsed_ms}ms)\n```\n{truncated}\n```"
                            continue
                    except Exception as e:
                        logger.warning(f"[Pipeline] OCR failed for {att.name}: {e}")

            except Exception as e:
                logger.warning(f"[Pipeline] Attachment preprocessing failed for {att.name}: {e}")

    # NOTE: _maybe_web_search() removed in v5.0
    # Web search is now handled by the tool system (ReAct loop in MistralHandler)
