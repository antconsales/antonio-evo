"""
Runtime Profiles for Antonio Evo Proto-AGI (v4.0).

Defines hardware-aware runtime configurations:
- EVO-LITE: Minimal hardware, local light LLM + external fallback
- EVO-STANDARD: Local LLM primary, API fallback allowed
- EVO-FULL: Local-first, multi-LLM, minimal external use
- EVO-HYBRID: Explicit external-first by user choice

Per Proto-AGI Spec:
- You are hardware-aware
- You operate within a Runtime Profile
- You adapt behavior to profile constraints
- You assume finite memory, compute, reasoning depth

Philosophy: Adapt to capabilities, never assume unlimited resources.
"""

import os
import platform
import psutil
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RuntimeProfile(Enum):
    """Available runtime profiles."""
    EVO_LITE = "evo-lite"        # Minimal hardware (RPi, <4GB RAM)
    EVO_STANDARD = "evo-standard"  # Standard setup (8-16GB RAM)
    EVO_FULL = "evo-full"        # Full local capabilities (16GB+ RAM)
    EVO_HYBRID = "evo-hybrid"    # External-first by user choice


@dataclass
class HardwareProfile:
    """Detected hardware capabilities."""
    total_ram_gb: float
    available_ram_gb: float
    cpu_cores: int
    cpu_freq_mhz: Optional[float]
    has_gpu: bool = False
    gpu_vram_gb: Optional[float] = None
    platform: str = ""
    is_low_power: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_ram_gb": round(self.total_ram_gb, 1),
            "available_ram_gb": round(self.available_ram_gb, 1),
            "cpu_cores": self.cpu_cores,
            "cpu_freq_mhz": round(self.cpu_freq_mhz, 0) if self.cpu_freq_mhz else None,
            "has_gpu": self.has_gpu,
            "gpu_vram_gb": self.gpu_vram_gb,
            "platform": self.platform,
            "is_low_power": self.is_low_power,
        }


@dataclass
class CognitiveBudgetParams:
    """
    Cognitive budget parameters for a runtime profile.

    Per Proto-AGI Spec:
    - max_reasoning_depth: Maximum chain-of-thought steps
    - max_context_tokens: Available context window
    - abstraction_level: Allowed abstraction complexity
    - simulation_budget: Max internal simulations
    - external_allowed: Can request external processing
    """
    max_reasoning_depth: int = 5
    max_context_tokens: int = 4096
    abstraction_level: str = "medium"  # low, medium, high, unlimited
    simulation_budget: int = 3
    external_allowed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_reasoning_depth": self.max_reasoning_depth,
            "max_context_tokens": self.max_context_tokens,
            "abstraction_level": self.abstraction_level,
            "simulation_budget": self.simulation_budget,
            "external_allowed": self.external_allowed,
        }


@dataclass
class ProfileCapabilities:
    """Capabilities available in a runtime profile."""
    # LLM capabilities
    local_llm_enabled: bool = True
    local_llm_max_context: int = 4096
    local_llm_models: List[str] = field(default_factory=list)
    external_llm_fallback: bool = False
    external_llm_primary: bool = False

    # Memory capabilities
    memory_enabled: bool = True
    emotional_memory: bool = True
    proactive_mode: bool = True
    personality_evolution: bool = True
    digital_twin: bool = True

    # Media capabilities
    image_generation: bool = False
    image_analysis: bool = False
    voice_input: bool = True
    voice_output: bool = True

    # RAG capabilities
    rag_enabled: bool = False
    rag_max_docs: int = 100

    # Response constraints
    max_response_tokens: int = 1024
    default_verbosity: str = "standard"  # minimal, standard, detailed

    # Cognitive budget (v4.0)
    cognitive_budget: CognitiveBudgetParams = field(default_factory=CognitiveBudgetParams)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "local_llm_enabled": self.local_llm_enabled,
            "local_llm_max_context": self.local_llm_max_context,
            "local_llm_models": self.local_llm_models,
            "external_llm_fallback": self.external_llm_fallback,
            "external_llm_primary": self.external_llm_primary,
            "memory_enabled": self.memory_enabled,
            "emotional_memory": self.emotional_memory,
            "proactive_mode": self.proactive_mode,
            "personality_evolution": self.personality_evolution,
            "digital_twin": self.digital_twin,
            "image_generation": self.image_generation,
            "image_analysis": self.image_analysis,
            "voice_input": self.voice_input,
            "voice_output": self.voice_output,
            "rag_enabled": self.rag_enabled,
            "rag_max_docs": self.rag_max_docs,
            "max_response_tokens": self.max_response_tokens,
            "default_verbosity": self.default_verbosity,
            "cognitive_budget": self.cognitive_budget.to_dict(),
        }


# Profile definitions with cognitive budgets (v4.0)
PROFILE_CAPABILITIES = {
    RuntimeProfile.EVO_LITE: ProfileCapabilities(
        local_llm_enabled=True,
        local_llm_max_context=2048,
        local_llm_models=["phi3:mini", "tinyllama", "qwen2.5:3b"],
        external_llm_fallback=True,
        external_llm_primary=False,
        memory_enabled=True,
        emotional_memory=True,
        proactive_mode=False,  # Disabled to save resources
        personality_evolution=True,
        digital_twin=False,  # Disabled to save resources
        image_generation=False,
        image_analysis=False,
        voice_input=True,
        voice_output=True,
        rag_enabled=False,
        max_response_tokens=512,
        default_verbosity="minimal",
        cognitive_budget=CognitiveBudgetParams(
            max_reasoning_depth=3,
            max_context_tokens=2048,
            abstraction_level="low",
            simulation_budget=0,  # No simulations on lite
            external_allowed=False,
        ),
    ),
    RuntimeProfile.EVO_STANDARD: ProfileCapabilities(
        local_llm_enabled=True,
        local_llm_max_context=4096,
        local_llm_models=["mistral", "qwen2.5:7b", "llama3.1:8b"],
        external_llm_fallback=True,
        external_llm_primary=False,
        memory_enabled=True,
        emotional_memory=True,
        proactive_mode=True,
        personality_evolution=True,
        digital_twin=True,
        image_generation=False,  # Too resource intensive
        image_analysis=True,
        voice_input=True,
        voice_output=True,
        rag_enabled=True,
        rag_max_docs=500,
        max_response_tokens=1024,
        default_verbosity="standard",
        cognitive_budget=CognitiveBudgetParams(
            max_reasoning_depth=5,
            max_context_tokens=4096,
            abstraction_level="medium",
            simulation_budget=3,
            external_allowed=False,
        ),
    ),
    RuntimeProfile.EVO_FULL: ProfileCapabilities(
        local_llm_enabled=True,
        local_llm_max_context=8192,
        local_llm_models=["mistral-nemo", "qwen2.5:14b", "llama3.1:70b"],
        external_llm_fallback=False,  # Local-first
        external_llm_primary=False,
        memory_enabled=True,
        emotional_memory=True,
        proactive_mode=True,
        personality_evolution=True,
        digital_twin=True,
        image_generation=True,
        image_analysis=True,
        voice_input=True,
        voice_output=True,
        rag_enabled=True,
        rag_max_docs=2000,
        max_response_tokens=2048,
        default_verbosity="detailed",
        cognitive_budget=CognitiveBudgetParams(
            max_reasoning_depth=10,
            max_context_tokens=8192,
            abstraction_level="high",
            simulation_budget=10,
            external_allowed=False,  # Still local-first
        ),
    ),
    RuntimeProfile.EVO_HYBRID: ProfileCapabilities(
        local_llm_enabled=True,
        local_llm_max_context=4096,
        local_llm_models=["mistral", "qwen2.5:7b"],
        external_llm_fallback=True,
        external_llm_primary=True,  # External preferred
        memory_enabled=True,
        emotional_memory=True,
        proactive_mode=True,
        personality_evolution=True,
        digital_twin=True,
        image_generation=True,
        image_analysis=True,
        voice_input=True,
        voice_output=True,
        rag_enabled=True,
        rag_max_docs=1000,
        max_response_tokens=2048,
        default_verbosity="standard",
        cognitive_budget=CognitiveBudgetParams(
            max_reasoning_depth=10,
            max_context_tokens=8192,
            abstraction_level="unlimited",
            simulation_budget=10,
            external_allowed=True,  # Network-enabled
        ),
    ),
}


class RuntimeProfileManager:
    """
    Manages runtime profile detection and selection.

    Detects hardware capabilities and selects appropriate profile.
    Can be overridden by user configuration.
    """

    def __init__(self, config_override: Optional[str] = None):
        """
        Initialize profile manager.

        Args:
            config_override: Optional profile name to force
        """
        self.hardware = self._detect_hardware()
        self.override = config_override
        self._active_profile: Optional[RuntimeProfile] = None
        self._capabilities: Optional[ProfileCapabilities] = None

    def _detect_hardware(self) -> HardwareProfile:
        """Detect system hardware capabilities."""
        try:
            mem = psutil.virtual_memory()
            total_ram = mem.total / (1024 ** 3)  # GB
            available_ram = mem.available / (1024 ** 3)  # GB

            cpu_count = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1
            cpu_freq = psutil.cpu_freq()
            cpu_mhz = cpu_freq.current if cpu_freq else None

            # Detect if low-power device (RPi, etc.)
            is_low_power = False
            if platform.machine() in ["armv7l", "aarch64"]:
                is_low_power = True
            elif total_ram < 4:
                is_low_power = True

            # Try to detect GPU (basic check)
            has_gpu = False
            gpu_vram = None
            try:
                import torch
                if torch.cuda.is_available():
                    has_gpu = True
                    gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            except ImportError:
                pass

            return HardwareProfile(
                total_ram_gb=total_ram,
                available_ram_gb=available_ram,
                cpu_cores=cpu_count,
                cpu_freq_mhz=cpu_mhz,
                has_gpu=has_gpu,
                gpu_vram_gb=gpu_vram,
                platform=platform.system(),
                is_low_power=is_low_power,
            )

        except Exception as e:
            logger.warning(f"Hardware detection failed: {e}")
            return HardwareProfile(
                total_ram_gb=8.0,  # Assume standard
                available_ram_gb=4.0,
                cpu_cores=4,
                cpu_freq_mhz=2000,
                platform=platform.system(),
            )

    def select_profile(self) -> RuntimeProfile:
        """
        Select appropriate runtime profile based on hardware.

        Returns:
            Selected RuntimeProfile
        """
        # Check for override
        if self.override:
            try:
                profile = RuntimeProfile(self.override.lower().replace("_", "-"))
                logger.info(f"Using overridden profile: {profile.value}")
                return profile
            except ValueError:
                logger.warning(f"Invalid profile override: {self.override}")

        # Check environment variable
        env_profile = os.environ.get("ANTONIO_PROFILE")
        if env_profile:
            try:
                profile = RuntimeProfile(env_profile.lower().replace("_", "-"))
                logger.info(f"Using environment profile: {profile.value}")
                return profile
            except ValueError:
                logger.warning(f"Invalid environment profile: {env_profile}")

        # Auto-detect based on hardware
        hw = self.hardware

        if hw.is_low_power or hw.total_ram_gb < 4:
            profile = RuntimeProfile.EVO_LITE
        elif hw.total_ram_gb < 16:
            profile = RuntimeProfile.EVO_STANDARD
        else:
            profile = RuntimeProfile.EVO_FULL

        logger.info(f"Auto-selected profile: {profile.value} (RAM: {hw.total_ram_gb:.1f}GB)")
        return profile

    def get_active_profile(self) -> RuntimeProfile:
        """Get the currently active profile."""
        if self._active_profile is None:
            self._active_profile = self.select_profile()
        return self._active_profile

    def get_capabilities(self) -> ProfileCapabilities:
        """Get capabilities for the active profile."""
        if self._capabilities is None:
            profile = self.get_active_profile()
            self._capabilities = PROFILE_CAPABILITIES[profile]
        return self._capabilities

    def is_capability_available(self, capability: str) -> bool:
        """
        Check if a specific capability is available.

        Args:
            capability: Capability name (e.g., "image_generation")

        Returns:
            True if available, False otherwise
        """
        caps = self.get_capabilities()
        return getattr(caps, capability, False)

    def get_response_constraints(self) -> Dict[str, Any]:
        """Get response generation constraints for the active profile."""
        caps = self.get_capabilities()
        return {
            "max_tokens": caps.max_response_tokens,
            "verbosity": caps.default_verbosity,
            "context_limit": caps.local_llm_max_context,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get runtime profile statistics."""
        profile = self.get_active_profile()
        caps = self.get_capabilities()

        return {
            "active_profile": profile.value,
            "hardware": self.hardware.to_dict(),
            "capabilities": caps.to_dict(),
            "constraints": self.get_response_constraints(),
        }

    def switch_profile(self, profile_name: str) -> bool:
        """
        Switch to a different profile.

        Args:
            profile_name: Name of profile to switch to

        Returns:
            True if successful, False otherwise
        """
        try:
            new_profile = RuntimeProfile(profile_name.lower().replace("_", "-"))
            self._active_profile = new_profile
            self._capabilities = PROFILE_CAPABILITIES[new_profile]
            logger.info(f"Switched to profile: {new_profile.value}")
            return True
        except ValueError:
            logger.error(f"Invalid profile name: {profile_name}")
            return False

    def get_cognitive_budget_params(self) -> CognitiveBudgetParams:
        """
        Get cognitive budget parameters for the active profile.

        Per Proto-AGI Spec:
        - Budget is tied to runtime profile
        - Budget must be respected at all times
        """
        caps = self.get_capabilities()
        return caps.cognitive_budget

    def create_cognitive_budget(self):
        """
        Create a CognitiveBudget instance for the active profile.

        Returns a budget configured for the current runtime profile.
        """
        try:
            from ..reasoning.cognitive_budget import CognitiveBudget, AbstractionLevel

            params = self.get_cognitive_budget_params()

            # Map abstraction level string to enum
            level_map = {
                "low": AbstractionLevel.LOW,
                "medium": AbstractionLevel.MEDIUM,
                "high": AbstractionLevel.HIGH,
                "unlimited": AbstractionLevel.UNLIMITED,
            }
            abstraction = level_map.get(params.abstraction_level, AbstractionLevel.MEDIUM)

            return CognitiveBudget(
                max_reasoning_depth=params.max_reasoning_depth,
                max_context_tokens=params.max_context_tokens,
                abstraction_level=abstraction,
                simulation_budget=params.simulation_budget,
                external_allowed=params.external_allowed,
            )
        except ImportError:
            logger.warning("Cognitive budget module not available")
            return None


# Singleton instance
_profile_manager: Optional[RuntimeProfileManager] = None


def get_profile_manager(config_override: Optional[str] = None) -> RuntimeProfileManager:
    """Get or create the profile manager singleton."""
    global _profile_manager
    if _profile_manager is None:
        _profile_manager = RuntimeProfileManager(config_override)
    return _profile_manager
