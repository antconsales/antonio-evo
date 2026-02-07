"""
Antonio Evo - Main Entry Point

This is the ONLY entry point.
No background processes. No hidden loops. No magic.

Now with:
- EvoMemory: Evolutionary memory that learns from interactions
- Dual Persona: SOCIAL (conversational) and LOGIC (analytical) modes
- HTTP-based handlers for faster ASR/TTS (optimized for Raspberry Pi)
- Ollama warm-up for reduced first-response latency
- RAG support for local knowledge base

Usage:
    python -m src.main                  # Interactive mode
    python -m src.main "your question"  # Single query mode
"""

import json
import time
import sys
import uuid
import logging
from typing import Dict, Any, Union, Optional

from .input.normalizer import Normalizer
from .policy.classifier import Classifier
from .policy.policy_engine import PolicyEngine
from .router.router import Router
from .output.response_builder import ResponseBuilder
from .utils.audit import AuditLogger
from .models.request import Request

# EvoMemory imports
from .memory.storage import MemoryStorage
from .memory.retriever import MemoryRetriever
from .memory.emotional import EmotionalMemory, EmotionalContext, ToneRecommendation
from .learning.neuron_creator import NeuronCreator

# Config loader
from .config.env_loader import get_config, ServiceConfig

# Services
from .services.ollama_warmup import OllamaWarmup
from .services.proactive import PatternAnalyzer, ProactiveService
from .services.personality import PersonalityEvolutionEngine, FeedbackSignal, PersonalityTrait
from .services.digital_twin import DigitalTwin
from .services.llm_manager import LLMManager, get_llm_manager

# Runtime Profiles
from .runtime import RuntimeProfileManager, RuntimeProfile, get_profile_manager

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Main orchestrator for Antonio Evo.

    Flow (deterministic, no exceptions):
    1. Normalize input -> Request
    2. Retrieve memory context -> MemoryContext
    3. Classify intent (memory-informed) -> Classification
    4. Apply policy -> PolicyDecision
    5. Route to handler -> Response
    6. Create neuron (if successful) -> Neuron
    7. Build response with metadata
    8. Log to audit trail (with memory operation)
    """

    def __init__(self, config_dir: str = "config"):
        """
        Initialize orchestrator.

        All components are initialized here.
        No lazy loading, no hidden state.
        """
        # Runtime Profile Manager - detect hardware and select profile
        self.profile_manager = get_profile_manager()
        self.active_profile = self.profile_manager.get_active_profile()
        self.profile_capabilities = self.profile_manager.get_capabilities()
        logger.info(f"Runtime Profile: {self.active_profile.value}")
        logger.info(f"Hardware: {self.profile_manager.hardware.total_ram_gb:.1f}GB RAM, "
                   f"{self.profile_manager.hardware.cpu_cores} cores, "
                   f"GPU: {self.profile_manager.hardware.has_gpu}")

        # Load environment configuration
        self.service_config = get_config()

        # Load memory config
        memory_config = self._load_config(f"{config_dir}/memory.json")

        # Core components
        self.normalizer = Normalizer()
        self.classifier = Classifier()
        self.policy = PolicyEngine(f"{config_dir}/policy.json")
        self.router = Router(f"{config_dir}/handlers.json", self.service_config)
        self.response_builder = ResponseBuilder()
        self.audit = AuditLogger()

        # EvoMemory components
        self.memory_enabled = memory_config.get("enabled", True)
        if self.memory_enabled:
            db_path = memory_config.get("database_path", "data/evomemory.db")
            self.memory_storage = MemoryStorage(db_path)
            self.memory_retriever = MemoryRetriever(
                self.memory_storage,
                memory_config.get("retrieval", {})
            )
            self.neuron_creator = NeuronCreator(
                self.memory_storage,
                memory_config.get("storage", {})
            )
            # v2.1 Emotional Memory
            self.emotional_memory = EmotionalMemory(db_path)
            logger.info("Emotional Memory (v2.1) initialized")

            # v2.2 Proactive Mode (respects profile capabilities)
            proactive_enabled = (
                memory_config.get("proactive_enabled", True) and
                self.profile_capabilities.proactive_mode
            )
            if proactive_enabled:
                self.pattern_analyzer = PatternAnalyzer(db_path)
                self.proactive_service = ProactiveService(
                    analyzer=self.pattern_analyzer,
                    analysis_interval=3600,  # Analyze patterns every hour
                    enabled=True,
                )
                self.proactive_service.start()
                logger.info("Proactive Mode (v2.2) initialized")
            else:
                self.pattern_analyzer = None
                self.proactive_service = None
                logger.info("Proactive Mode (v2.2) disabled by profile")

            # v2.3 Personality Evolution
            self.personality_engine = PersonalityEvolutionEngine(db_path)
            logger.info("Personality Evolution (v2.3) initialized")

            # v3.0 Digital Twin (respects profile capabilities)
            if self.profile_capabilities.digital_twin:
                self.digital_twin = DigitalTwin(db_path)
                logger.info("Digital Twin (v3.0) initialized")
            else:
                self.digital_twin = None
                logger.info("Digital Twin (v3.0) disabled by profile")
        else:
            self.memory_storage = None
            self.memory_retriever = None
            self.neuron_creator = None
            self.emotional_memory = None
            self.pattern_analyzer = None
            self.proactive_service = None
            self.personality_engine = None
            self.digital_twin = None

        # RAG components (optional)
        self.rag = None
        if self.service_config.rag_enabled:
            self._init_rag()

        # Ollama warm-up (optional but recommended)
        self.warmup = None
        if self.service_config.ollama_warmup_enabled:
            self._init_warmup()

        # Multi-LLM Manager
        self.llm_manager = self._init_llm_manager()

        # Session management
        self.current_session_id: Optional[str] = None

    def _init_warmup(self) -> None:
        """Initialize Ollama warm-up service."""
        try:
            self.warmup = OllamaWarmup(
                base_url=self.service_config.llm_server,
                model=self.service_config.ollama_model,
                warmup_prompt=self.service_config.ollama_warmup_prompt,
                keepalive_minutes=self.service_config.ollama_keepalive_minutes
            )

            # Perform warm-up (blocking)
            logger.info("Starting Ollama warm-up...")
            if self.warmup.warmup(timeout=120):
                logger.info("Ollama warm-up complete")
                # Start keep-alive thread
                self.warmup.start_keepalive()
            else:
                logger.warning("Ollama warm-up failed, continuing without it")

        except Exception as e:
            logger.warning(f"Failed to initialize Ollama warm-up: {e}")
            self.warmup = None

    def _init_rag(self) -> None:
        """Initialize RAG components."""
        try:
            from .rag.qdrant_client import QdrantRAG

            rag_config = {
                "server_url": self.service_config.qdrant_server,
                "embedding_model": self.service_config.rag_embedding_model,
                "docs_path": self.service_config.rag_docs_path,
                "chunk_size": self.service_config.rag_chunk_size,
                "chunk_overlap": self.service_config.rag_chunk_overlap,
                "top_k": self.service_config.rag_top_k
            }

            self.rag = QdrantRAG(rag_config)

            if self.rag.is_available():
                logger.info("RAG initialized successfully")
            else:
                logger.warning("RAG initialized but not available (check Qdrant)")

        except ImportError as e:
            logger.warning(f"RAG dependencies not available: {e}")
            self.rag = None
        except Exception as e:
            logger.warning(f"Failed to initialize RAG: {e}")
            self.rag = None

    def _init_llm_manager(self) -> Optional[LLMManager]:
        """Initialize Multi-LLM Manager."""
        try:
            db_path = "data/evomemory.db"
            llm_manager = get_llm_manager(db_path, self.profile_capabilities)

            # Configure from environment
            llm_manager.configure_from_env(self.service_config)

            # Check availability
            availability = llm_manager.check_availability()
            available_count = sum(
                1 for status in availability.values()
                if status.value == "available"
            )

            logger.info(f"LLM Manager: {available_count}/{len(availability)} endpoints available")
            return llm_manager

        except Exception as e:
            logger.warning(f"Failed to initialize LLM Manager: {e}")
            return None

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load JSON config file, return empty dict if not found."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def start_session(self) -> str:
        """Start a new conversation session."""
        self.current_session_id = str(uuid.uuid4())[:12]
        if self.memory_storage:
            self.memory_storage.create_session(self.current_session_id)
        return self.current_session_id

    def end_session(self):
        """End the current session."""
        if self.current_session_id and self.memory_storage:
            self.memory_storage.end_session(self.current_session_id)
        self.current_session_id = None

    def process(self, raw_input: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a single request.

        SYNCHRONOUS. One request in, one response out.
        No async, no callbacks, no side effects beyond audit log and memory.
        """

        start_time = time.time()
        memory_operation = None

        # === STEP 1: Normalize ===
        norm_result = self.normalizer.normalize(raw_input)

        if not norm_result.success:
            # Normalization failed - return error response
            return {
                "success": False,
                "error": "; ".join([e.message for e in norm_result.errors]),
                "error_code": norm_result.error_code,
                "elapsed_ms": int((time.time() - start_time) * 1000),
            }

        request = norm_result.request

        # Attach session ID
        request.session_id = self.current_session_id

        # === STEP 2: Memory Retrieval ===
        if self.memory_enabled and self.memory_retriever:
            try:
                memory_context = self.memory_retriever.retrieve(
                    query=request.text,
                    session_id=request.session_id,
                )
                request.memory_context = memory_context
            except Exception:
                # Memory retrieval failed, continue without it
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
                # Emotional analysis failed, continue without it
                request.emotional_context = None

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

                # Create memory operation for audit
                memory_operation = self.neuron_creator.create_memory_operation(
                    request=request,
                    neuron=neuron,
                )

                # Add neuron info to response metadata
                if neuron:
                    response["_meta"] = response.get("_meta", {})
                    response["_meta"]["neuron_stored"] = True
                    response["_meta"]["neuron_id"] = neuron.id
                    response["_meta"]["neuron_confidence"] = neuron.confidence

            except Exception:
                # Neuron creation failed, continue without it
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
                # Non-critical, continue
                pass

        # === STEP 7.3: Learn User Style for Digital Twin (v3.0) ===
        if self.digital_twin:
            try:
                self.digital_twin.learn_from_message(request.text)
            except Exception:
                # Non-critical, continue
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

        return response

    def health_check(self) -> Dict[str, Any]:
        """
        Check system health.

        Returns status of all components including memory stats.
        """
        result = {
            "status": "ok",
            "version": "2.0-evo",
            "handlers": self.router.get_available_handlers(),
            "timestamp": time.time(),
            "session_id": self.current_session_id,
        }

        # Add runtime profile info
        result["profile"] = self.profile_manager.get_stats()

        # Add memory stats if enabled
        if self.memory_enabled and self.memory_storage:
            try:
                memory_stats = self.memory_storage.get_stats()
                result["memory"] = {
                    "enabled": True,
                    "total_neurons": memory_stats.get("total_neurons", 0),
                    "avg_confidence": round(memory_stats.get("avg_confidence", 0), 3),
                    "total_accesses": memory_stats.get("total_accesses", 0),
                }
            except Exception:
                result["memory"] = {"enabled": True, "error": "Failed to get stats"}
        else:
            result["memory"] = {"enabled": False}

        # Add warmup status
        if self.warmup:
            result["warmup"] = self.warmup.get_status()
        else:
            result["warmup"] = {"enabled": False}

        # Add RAG status
        if self.rag:
            result["rag"] = self.rag.get_stats()
        else:
            result["rag"] = {"enabled": False}

        # Add Emotional Memory status (v2.1)
        if self.emotional_memory:
            try:
                emotional_stats = self.emotional_memory.get_stats()
                result["emotional_memory"] = {
                    "enabled": True,
                    "version": "2.1",
                    "total_signals": emotional_stats.get("total_signals", 0),
                    "weekly_distribution": emotional_stats.get("weekly_distribution", {}),
                    "avg_confidence": emotional_stats.get("avg_confidence", 0),
                }
            except Exception:
                result["emotional_memory"] = {"enabled": True, "error": "Failed to get stats"}
        else:
            result["emotional_memory"] = {"enabled": False}

        # Add Proactive Mode status (v2.2)
        if self.pattern_analyzer:
            try:
                proactive_stats = self.pattern_analyzer.get_stats()
                result["proactive"] = proactive_stats
            except Exception:
                result["proactive"] = {"enabled": True, "error": "Failed to get stats"}
        else:
            result["proactive"] = {"enabled": False}

        # Add Personality Evolution status (v2.3)
        if self.personality_engine:
            try:
                personality_stats = self.personality_engine.get_stats()
                result["personality"] = personality_stats
            except Exception:
                result["personality"] = {"enabled": True, "error": "Failed to get stats"}
        else:
            result["personality"] = {"enabled": False}

        # Add Digital Twin status (v3.0)
        if self.digital_twin:
            try:
                twin_stats = self.digital_twin.get_stats()
                result["digital_twin"] = twin_stats
            except Exception:
                result["digital_twin"] = {"enabled": True, "error": "Failed to get stats"}
        else:
            result["digital_twin"] = {"enabled": False}

        # Add LLM Manager status
        if self.llm_manager:
            try:
                llm_stats = self.llm_manager.get_stats()
                result["llm_manager"] = llm_stats
            except Exception:
                result["llm_manager"] = {"enabled": True, "error": "Failed to get stats"}
        else:
            result["llm_manager"] = {"enabled": False}

        return result

    def memory_search(self, query: str, limit: int = 5) -> list:
        """Search memory for similar past interactions."""
        if not self.memory_enabled or not self.memory_retriever:
            return []

        results = self.memory_retriever.find_similar(query, limit=limit)
        return [
            {
                "id": rn.neuron.id,
                "input": rn.neuron.input[:100] + "..." if len(rn.neuron.input) > 100 else rn.neuron.input,
                "confidence": rn.neuron.confidence,
                "relevance": round(rn.relevance_score, 3),
                "persona": rn.neuron.persona.value,
            }
            for rn in results
        ]


def main():
    """CLI entry point."""

    orchestrator = Orchestrator()

    # Single command mode
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        orchestrator.start_session()
        result = orchestrator.process(text)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        orchestrator.end_session()
        return

    # Interactive mode - start session
    session_id = orchestrator.start_session()

    print("=" * 50)
    print("  ANTONIO EVO")
    print("  Local-first | Evolutionary | Auditable")
    print("=" * 50)
    print()
    print(f"Session: {session_id}")
    print()
    print("Commands:")
    print("  /health     - Check system status")
    print("  /profile    - Show runtime profile and capabilities")
    print("  /llm        - Show LLM endpoints and status")
    print("  /recent     - Show recent audit logs")
    print("  /memory     - Show memory stats")
    print("  /emotion    - Show emotional context stats (v2.1)")
    print("  /insights   - Show proactive insights (v2.2)")
    print("  /persona    - Show personality traits (v2.3)")
    print("  /twin       - Show digital twin status (v3.0)")
    print("  /search <q> - Search memory")
    print("  /exit       - Exit")
    print()

    while True:
        try:
            user_input = input("> ").strip()

            if not user_input:
                continue

            # Built-in commands
            if user_input == "/exit":
                print("Saving session...")
                orchestrator.end_session()
                print("Exiting...")
                break

            if user_input == "/health":
                print(json.dumps(orchestrator.health_check(), indent=2))
                continue

            if user_input == "/profile":
                stats = orchestrator.profile_manager.get_stats()
                profile = stats["active_profile"]
                hw = stats["hardware"]
                caps = stats["capabilities"]

                print(f"Runtime Profile: {profile.upper()}")
                print()
                print("Hardware Detected:")
                print(f"  RAM: {hw['total_ram_gb']}GB total, {hw['available_ram_gb']}GB available")
                print(f"  CPU: {hw['cpu_cores']} cores @ {hw['cpu_freq_mhz']}MHz")
                print(f"  GPU: {'Yes (' + str(hw['gpu_vram_gb']) + 'GB VRAM)' if hw['has_gpu'] else 'No'}")
                print(f"  Platform: {hw['platform']}")
                print(f"  Low Power: {'Yes' if hw['is_low_power'] else 'No'}")
                print()
                print("Capabilities:")
                print(f"  Local LLM: {caps['local_llm_enabled']} (max {caps['local_llm_max_context']} context)")
                print(f"  Recommended models: {', '.join(caps['local_llm_models'][:2])}")
                print(f"  External fallback: {caps['external_llm_fallback']}")
                print(f"  Image generation: {caps['image_generation']}")
                print(f"  Image analysis: {caps['image_analysis']}")
                print(f"  RAG: {caps['rag_enabled']} (max {caps['rag_max_docs']} docs)")
                print(f"  Proactive mode: {caps['proactive_mode']}")
                print(f"  Digital twin: {caps['digital_twin']}")
                print()
                print(f"Response: max {caps['max_response_tokens']} tokens, {caps['default_verbosity']} verbosity")
                continue

            if user_input == "/llm":
                if orchestrator.llm_manager:
                    models = orchestrator.llm_manager.get_available_models()
                    print("LLM Endpoints:")
                    print()
                    for model in models:
                        status_icon = {
                            "available": "✓",
                            "unavailable": "✗",
                            "error": "!",
                            "rate_limited": "~",
                            "warming_up": "...",
                        }.get(model["status"], "?")
                        local_str = "LOCAL" if model["is_local"] else "CLOUD"
                        print(f"  [{status_icon}] {model['provider']}:{model['name']} ({local_str})")
                        print(f"      Model: {model['model']}")
                        print(f"      Status: {model['status']}")
                        if model["avg_latency_ms"] > 0:
                            print(f"      Avg latency: {model['avg_latency_ms']}ms")
                        print()

                    # Show 24h stats
                    stats = orchestrator.llm_manager.get_stats()
                    provider_stats = stats.get("provider_stats_24h", {})
                    if provider_stats:
                        print("Last 24h stats:")
                        for provider, pstats in provider_stats.items():
                            success_pct = pstats["success_rate"] * 100
                            print(f"  {provider}: {pstats['total_requests']} requests, "
                                  f"{success_pct:.0f}% success, "
                                  f"{pstats['avg_latency_ms']}ms avg")
                else:
                    print("LLM Manager not initialized")
                continue

            if user_input == "/recent":
                recent = orchestrator.audit.get_recent(5)
                for entry in recent:
                    handler = entry.get("decision", {}).get("handler", "?")
                    persona = entry.get("decision", {}).get("persona", "?")
                    elapsed = entry.get("elapsed_ms", 0)
                    neuron = entry.get("memory_operation", {}).get("stored_neuron_id", "")
                    neuron_str = f" [+{neuron[:6]}]" if neuron else ""
                    print(f"[{entry.get('timestamp_iso', '?')}] "
                          f"{handler}/{persona} ({elapsed}ms){neuron_str}")
                continue

            if user_input == "/memory":
                health = orchestrator.health_check()
                memory = health.get("memory", {})
                if memory.get("enabled"):
                    print(f"Neurons: {memory.get('total_neurons', 0)}")
                    print(f"Avg confidence: {memory.get('avg_confidence', 0)}")
                    print(f"Total accesses: {memory.get('total_accesses', 0)}")
                else:
                    print("Memory disabled")
                continue

            if user_input == "/emotion":
                health = orchestrator.health_check()
                emo = health.get("emotional_memory", {})
                if emo.get("enabled"):
                    print(f"Emotional Memory v{emo.get('version', '2.1')}")
                    print(f"Total signals recorded: {emo.get('total_signals', 0)}")
                    print(f"Average confidence: {emo.get('avg_confidence', 0):.2f}")
                    dist = emo.get("weekly_distribution", {})
                    if dist:
                        print("Weekly distribution:")
                        for state, count in sorted(dist.items(), key=lambda x: -x[1])[:5]:
                            print(f"  {state}: {count}")
                else:
                    print("Emotional memory disabled")
                continue

            if user_input == "/insights":
                if orchestrator.pattern_analyzer:
                    # Get pending insights
                    insights = orchestrator.pattern_analyzer.get_pending_insights(limit=5)
                    if insights:
                        print("Proactive Insights (v2.2):")
                        for insight in insights:
                            priority_icon = {"high": "!", "medium": "*", "low": "-"}.get(insight.priority.value, "-")
                            print(f"  [{priority_icon}] {insight.message}")
                            orchestrator.pattern_analyzer.mark_insight_shown(insight.id)
                    else:
                        # No pending insights, show stats
                        stats = orchestrator.pattern_analyzer.get_stats()
                        print(f"Proactive Mode v{stats.get('version', '2.2')}")
                        print(f"Interactions logged: {stats.get('total_interactions', 0)}")
                        print(f"Patterns detected: {stats.get('patterns_detected', 0)}")
                        print("No new insights at this time.")
                else:
                    print("Proactive mode disabled")
                continue

            if user_input == "/persona":
                if orchestrator.personality_engine:
                    profile = orchestrator.personality_engine.get_profile()
                    stats = orchestrator.personality_engine.get_stats()
                    print(f"Personality Evolution v{stats.get('version', '2.3')}")
                    print(f"Total evolutions: {stats.get('total_evolutions', 0)}")
                    print("\nCurrent traits (0-100 scale):")
                    for trait, value in sorted(stats.get('current_profile', {}).items()):
                        bar = "#" * (value // 10) + "-" * (10 - value // 10)
                        print(f"  {trait:12}: [{bar}] {value}")
                    print("\nResponse guidelines:")
                    for key, guideline in profile.get_response_guidelines().items():
                        print(f"  {key}: {guideline}")
                else:
                    print("Personality evolution disabled")
                continue

            if user_input == "/twin":
                if orchestrator.digital_twin:
                    stats = orchestrator.digital_twin.get_stats()
                    print(f"Digital Twin v{stats.get('version', '3.0')}")
                    print(f"Messages analyzed: {stats.get('messages_analyzed', 0)}/{stats.get('messages_needed', 50)}")
                    readiness = stats.get('readiness', 0)
                    ready_bar = "#" * int(readiness * 10) + "-" * (10 - int(readiness * 10))
                    print(f"Readiness: [{ready_bar}] {readiness * 100:.0f}%")
                    print(f"Vocabulary size: {stats.get('vocabulary_size', 0)} words")
                    print(f"Patterns learned: {stats.get('patterns_learned', 0)}")
                    print(f"Expressions captured: {stats.get('expressions_captured', 0)}")
                    print(f"\nStyle summary: {stats.get('style_summary', 'Not enough data')}")
                    if stats.get('ready'):
                        print("\n[READY] Digital Twin can now mimic your communication style!")
                else:
                    print("Digital twin disabled")
                continue

            if user_input.startswith("/search "):
                query = user_input[8:].strip()
                if query:
                    results = orchestrator.memory_search(query)
                    if results:
                        for r in results:
                            print(f"[{r['id']}] {r['input']}")
                            print(f"   confidence: {r['confidence']:.2f}, "
                                  f"relevance: {r['relevance']:.2f}, "
                                  f"persona: {r['persona']}")
                    else:
                        print("No matching neurons found")
                continue

            # Process request
            result = orchestrator.process(user_input)

            # Pretty print response
            print()
            if result.get("success"):
                output = result.get("output") or result.get("text", "")
                print(output)

                # Show metadata hints
                meta = result.get("_meta", {})
                hints = []
                if meta.get("neuron_stored"):
                    hints.append(f"[Learned: {meta.get('neuron_confidence', 0):.2f}]")
                if result.get("decision", {}).get("persona"):
                    hints.append(f"[{result['decision']['persona'].upper()}]")
                if hints:
                    print()
                    print(" ".join(hints))
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
            print()

        except KeyboardInterrupt:
            print("\nInterrupted. Saving session...")
            orchestrator.end_session()
            print("Exiting...")
            break
        except EOFError:
            orchestrator.end_session()
            break


if __name__ == "__main__":
    main()
