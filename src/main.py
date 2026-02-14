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

# Extracted components
from .session.session_manager import SessionManager
from .pipeline.executor import PipelineExecutor
from .health.monitor import HealthMonitor

# Tool system (v5.0)
from .tools import ToolRegistry, ToolExecutor

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Main orchestrator for Antonio Evo.

    Coordinates sub-components:
    - SessionManager: session lifecycle
    - PipelineExecutor: 8-step deterministic processing pipeline
    - HealthMonitor: system health reporting

    Maintains backward compatibility with the same public API.
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
        normalizer = Normalizer()
        classifier = Classifier(
            ollama_url=self.service_config.llm_server,
            model=self.service_config.ollama_model
        )
        policy = PolicyEngine(f"{config_dir}/policy.json")
        self.router = Router(f"{config_dir}/handlers.json", self.service_config)
        response_builder = ResponseBuilder()
        audit = AuditLogger()

        # Store refs needed by CLI commands and websocket_server
        self.audit = audit
        self.classifier = classifier

        # EvoMemory components
        self.memory_enabled = memory_config.get("enabled", True)
        self.memory_storage = None
        self.memory_retriever = None
        self.neuron_creator = None
        self.emotional_memory = None
        self.pattern_analyzer = None
        self.proactive_service = None
        self.personality_engine = None
        self.digital_twin = None

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
                    analysis_interval=3600,
                    enabled=True,
                )
                self.proactive_service.start()
                logger.info("Proactive Mode (v2.2) initialized")
            else:
                logger.info("Proactive Mode (v2.2) disabled by profile")

            # v2.3 Personality Evolution
            self.personality_engine = PersonalityEvolutionEngine(db_path)
            logger.info("Personality Evolution (v2.3) initialized")

            # v3.0 Digital Twin (respects profile capabilities)
            if self.profile_capabilities.digital_twin:
                self.digital_twin = DigitalTwin(db_path)
                logger.info("Digital Twin (v3.0) initialized")
            else:
                logger.info("Digital Twin (v3.0) disabled by profile")

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

        # --- Extracted components ---

        # Session Manager
        self.session_manager = SessionManager(self.memory_storage)

        # Pipeline Executor (all dependencies injected)
        # Webhook Service (n8n integration)
        self.webhook_service = self._init_webhook_service()

        # Web Search Service (Tavily)
        self.web_search_service = self._init_web_search()

        # Tool System (v5.0 - Agentic Tool Use)
        self.tool_registry = None
        self.tool_executor = None
        self._init_tools()

        self.pipeline = PipelineExecutor(
            normalizer=normalizer,
            classifier=classifier,
            policy=policy,
            router=self.router,
            response_builder=response_builder,
            audit=audit,
            session_manager=self.session_manager,
            memory_enabled=self.memory_enabled,
            memory_retriever=self.memory_retriever,
            neuron_creator=self.neuron_creator,
            emotional_memory=self.emotional_memory,
            proactive_service=self.proactive_service,
            digital_twin=self.digital_twin,
            webhook_service=self.webhook_service,
            web_search_service=self.web_search_service,
        )

        # Health Monitor (all dependencies injected)
        self.health_monitor = HealthMonitor(
            router=self.router,
            profile_manager=self.profile_manager,
            session_manager=self.session_manager,
            memory_enabled=self.memory_enabled,
            memory_storage=self.memory_storage,
            warmup=self.warmup,
            rag=self.rag,
            emotional_memory=self.emotional_memory,
            pattern_analyzer=self.pattern_analyzer,
            personality_engine=self.personality_engine,
            digital_twin=self.digital_twin,
            llm_manager=self.llm_manager,
        )

    def _init_warmup(self) -> None:
        """Initialize Ollama warm-up service."""
        try:
            self.warmup = OllamaWarmup(
                base_url=self.service_config.llm_server,
                model=self.service_config.ollama_model,
                warmup_prompt=self.service_config.ollama_warmup_prompt,
                keepalive_minutes=self.service_config.ollama_keepalive_minutes
            )

            logger.info("Starting Ollama warm-up...")
            if self.warmup.warmup(timeout=120):
                logger.info("Ollama warm-up complete")
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

            llm_manager.configure_from_env(self.service_config)

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

    def _init_webhook_service(self):
        """Initialize n8n Webhook Service."""
        try:
            from .services.webhook_service import get_webhook_service
            service = get_webhook_service()
            logger.info(f"Webhook Service initialized ({len(service.webhooks)} webhooks configured)")
            return service
        except Exception as e:
            logger.warning(f"Failed to initialize Webhook Service: {e}")
            return None

    def _register_tool(self, registry, definition: dict, handler):
        """Helper to register a tool from its DEFINITION dict and handler."""
        registry.register(
            name=definition["name"],
            description=definition["description"],
            parameters=definition["parameters"],
            handler=handler,
        )

    def _init_tools(self) -> None:
        """Initialize Tool System (v5.0) - registry, tools, executor."""
        try:
            registry = ToolRegistry()

            # Web Search tool (wraps existing Tavily service)
            from .tools.web_search import DEFINITION as ws_def, create_handler as ws_handler
            if self.web_search_service:
                self._register_tool(registry, ws_def, ws_handler(self.web_search_service))

            # File operations tools (read, write, list)
            from .tools.file_ops import DEFINITIONS as fo_defs, create_handlers as fo_handlers
            file_handlers = fo_handlers()
            for defn in fo_defs:
                handler = file_handlers.get(defn["name"])
                if handler:
                    self._register_tool(registry, defn, handler)

            # Code execution tool
            from .tools.code_exec import DEFINITION as ce_def, create_handler as ce_handler
            self._register_tool(registry, ce_def, ce_handler())

            # Image analysis tool (wraps existing VisionService)
            vision_service = getattr(self.router, 'vision_service', None)
            if vision_service:
                from .tools.image_analysis import DEFINITION as ia_def, create_handler as ia_handler
                self._register_tool(registry, ia_def, ia_handler(vision_service))

            self.tool_registry = registry
            self.tool_executor = ToolExecutor(registry)

            # Inject tools into text handlers
            self._inject_tools_into_handlers()

            logger.info(f"Tool System (v5.0): {len(registry)} tools registered ({', '.join(registry.tool_names)})")

        except Exception as e:
            logger.warning(f"Failed to initialize Tool System: {e}")
            self.tool_registry = None
            self.tool_executor = None

    def _inject_tools_into_handlers(self) -> None:
        """Inject tool registry and executor into text handlers."""
        if not self.tool_registry or not self.tool_executor:
            return

        from .models.policy import Handler
        for handler_enum in (Handler.TEXT_LOCAL, Handler.TEXT_SOCIAL, Handler.TEXT_LOGIC):
            handler = self.router.handlers.get(handler_enum)
            if handler and hasattr(handler, 'set_tool_registry'):
                handler.set_tool_registry(self.tool_registry)
                handler.set_tool_executor(self.tool_executor)

    def _init_web_search(self):
        """Initialize Tavily Web Search Service."""
        try:
            from .services.web_search import get_web_search_service
            service = get_web_search_service()
            if service.is_available():
                logger.info("Web Search Service (Tavily) initialized")
            else:
                logger.info("Web Search Service configured but no API key set")
            return service
        except Exception as e:
            logger.warning(f"Failed to initialize Web Search Service: {e}")
            return None

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load JSON config file, return empty dict if not found."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    # === Public API (backward compatible) ===

    @property
    def current_session_id(self) -> Optional[str]:
        """Get current session ID."""
        return self.session_manager.current_session_id

    @current_session_id.setter
    def current_session_id(self, value: Optional[str]):
        """Set current session ID (used by websocket_server)."""
        self.session_manager.current_session_id = value

    def start_session(self) -> str:
        """Start a new conversation session."""
        return self.session_manager.start_session()

    def end_session(self):
        """End the current session."""
        self.session_manager.end_session()

    def process(self, raw_input: Union[str, Dict[str, Any]], tool_callback=None) -> Dict[str, Any]:
        """
        Process a single request.

        SYNCHRONOUS. One request in, one response out.
        Delegates to PipelineExecutor.

        Args:
            raw_input: User input (string or dict)
            tool_callback: Optional callback for real-time tool action events (v5.0)
        """
        return self.pipeline.execute(raw_input, tool_callback=tool_callback)

    def health_check(self) -> Dict[str, Any]:
        """
        Check system health.

        Delegates to HealthMonitor.
        """
        return self.health_monitor.check()

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
                    payload = entry.payload if hasattr(entry, 'payload') else entry
                    handler = payload.get("decision", {}).get("handler", "?")
                    persona = payload.get("decision", {}).get("persona", "?")
                    elapsed = payload.get("elapsed_ms", 0)
                    mem_op = payload.get("memory_operation") or {}
                    neuron = mem_op.get("stored_neuron_id", "")
                    neuron_str = f" [+{neuron[:6]}]" if neuron else ""
                    ts = entry.timestamp_iso if hasattr(entry, 'timestamp_iso') else "?"
                    print(f"[{ts}] "
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
                    insights = orchestrator.pattern_analyzer.get_pending_insights(limit=5)
                    if insights:
                        print("Proactive Insights (v2.2):")
                        for insight in insights:
                            priority_icon = {"high": "!", "medium": "*", "low": "-"}.get(insight.priority.value, "-")
                            print(f"  [{priority_icon}] {insight.message}")
                            orchestrator.pattern_analyzer.mark_insight_shown(insight.id)
                    else:
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
