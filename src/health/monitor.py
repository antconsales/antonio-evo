"""
Health Monitor - System health checking and status reporting.

Extracted from Orchestrator to follow Single Responsibility Principle.
"""

import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class HealthMonitor:
    """
    Monitors system health by querying all components.

    All components are injected - no hidden dependencies.
    """

    def __init__(
        self,
        router,
        profile_manager,
        session_manager,
        memory_enabled: bool = False,
        memory_storage=None,
        warmup=None,
        rag=None,
        emotional_memory=None,
        pattern_analyzer=None,
        personality_engine=None,
        digital_twin=None,
        llm_manager=None,
    ):
        self.router = router
        self.profile_manager = profile_manager
        self.session_manager = session_manager
        self.memory_enabled = memory_enabled
        self.memory_storage = memory_storage
        self.warmup = warmup
        self.rag = rag
        self.emotional_memory = emotional_memory
        self.pattern_analyzer = pattern_analyzer
        self.personality_engine = personality_engine
        self.digital_twin = digital_twin
        self.llm_manager = llm_manager

    def check(self) -> Dict[str, Any]:
        """
        Check system health.

        Returns status of all components including memory stats.
        """
        result = {
            "status": "ok",
            "version": "2.0-evo",
            "handlers": self.router.get_available_handlers(),
            "timestamp": time.time(),
            "session_id": self.session_manager.current_session_id,
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
