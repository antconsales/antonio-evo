"""
Environment Configuration Loader

Loads configuration from .env file with fallback to existing JSON configs.
Provides a single source of truth for all service URLs and settings.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path

# Try to import python-dotenv, graceful fallback if not available
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


@dataclass
class ServiceConfig:
    """Configuration for all external services."""

    # Service URLs
    asr_server: str = "http://localhost:8803"
    asr_timeout: int = 30

    tts_server: str = "http://localhost:8804"
    tts_timeout: int = 10

    llm_server: str = "http://localhost:11434"
    llm_timeout: int = 120

    qdrant_server: str = "http://localhost:6333"

    # Model selection
    ollama_model: str = "mistral"
    ollama_context_length: int = 4096

    whisper_model: str = "base"
    whisper_language: str = "auto"

    piper_voice: str = "it_IT-riccardo-x_low"
    piper_speed: float = 1.0

    # Warm-up settings
    ollama_warmup_enabled: bool = True
    ollama_warmup_prompt: str = "Ciao, come posso aiutarti?"
    ollama_keepalive_minutes: int = 5

    # RAG settings
    rag_enabled: bool = False
    rag_docs_path: str = "data/knowledge"
    rag_embedding_model: str = "all-MiniLM-L6-v2"
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 50
    rag_top_k: int = 3

    # Feature flags
    use_http_handlers: bool = True
    memory_enabled: bool = True
    audit_enabled: bool = True

    # Paths
    data_dir: str = "data"
    logs_dir: str = "logs"
    config_dir: str = "config"

    # External API (fallback)
    external_api_key: str = ""
    external_provider: str = "anthropic"
    external_model: str = "claude-3-haiku-20240307"


def _load_json_config(path: str) -> Dict[str, Any]:
    """Load JSON config file, return empty dict if not found."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _get_env(key: str, default: Any = None, cast_type: type = str) -> Any:
    """Get environment variable with type casting."""
    value = os.environ.get(key)
    if value is None:
        return default

    if cast_type == bool:
        return value.lower() in ("true", "1", "yes", "on")
    elif cast_type == int:
        try:
            return int(value)
        except ValueError:
            return default
    elif cast_type == float:
        try:
            return float(value)
        except ValueError:
            return default

    return value


def load_config(env_path: Optional[str] = None, config_dir: str = "config") -> ServiceConfig:
    """
    Load configuration from .env file with JSON fallback.

    Priority:
    1. Environment variables (highest)
    2. .env file
    3. JSON config files (lowest)

    Args:
        env_path: Path to .env file (default: project root/.env)
        config_dir: Path to config directory for JSON fallback

    Returns:
        ServiceConfig with all settings loaded
    """
    # Load .env file if available
    if DOTENV_AVAILABLE:
        if env_path:
            load_dotenv(env_path)
        else:
            # Try to find .env in project root
            project_root = Path(__file__).parent.parent.parent
            env_file = project_root / ".env"
            if env_file.exists():
                load_dotenv(env_file)

    # Load JSON configs for fallback
    handlers_json = _load_json_config(f"{config_dir}/handlers.json")
    voice_json = _load_json_config(f"{config_dir}/voice.json")
    memory_json = _load_json_config(f"{config_dir}/memory.json")

    # Extract nested values from JSON
    whisper_config = handlers_json.get("handlers", {}).get("whisper", {})
    tts_config = handlers_json.get("handlers", {}).get("tts", {})
    mistral_config = handlers_json.get("handlers", {}).get("mistral", {})

    # Build config with priority: env > .env > json > defaults
    config = ServiceConfig(
        # Service URLs
        asr_server=_get_env("ASR_SERVER",
            whisper_config.get("server_url", "http://localhost:8803")),
        asr_timeout=_get_env("ASR_TIMEOUT",
            whisper_config.get("timeout", 30), int),

        tts_server=_get_env("TTS_SERVER",
            tts_config.get("server_url", "http://localhost:8804")),
        tts_timeout=_get_env("TTS_TIMEOUT",
            tts_config.get("timeout", 10), int),

        llm_server=_get_env("LLM_SERVER",
            mistral_config.get("base_url", "http://localhost:11434")),
        llm_timeout=_get_env("LLM_TIMEOUT",
            mistral_config.get("timeout", 120), int),

        qdrant_server=_get_env("QDRANT_SERVER", "http://localhost:6333"),

        # Models
        ollama_model=_get_env("OLLAMA_MODEL",
            mistral_config.get("model", "mistral")),
        ollama_context_length=_get_env("OLLAMA_CONTEXT_LENGTH", 4096, int),

        whisper_model=_get_env("WHISPER_MODEL",
            voice_json.get("input", {}).get("model", "base")),
        whisper_language=_get_env("WHISPER_LANGUAGE",
            voice_json.get("input", {}).get("language", "auto")),

        piper_voice=_get_env("PIPER_VOICE",
            tts_config.get("voice", "it_IT-riccardo-x_low")),
        piper_speed=_get_env("PIPER_SPEED",
            voice_json.get("output", {}).get("speed", 1.0), float),

        # Warm-up
        ollama_warmup_enabled=_get_env("OLLAMA_WARMUP_ENABLED", True, bool),
        ollama_warmup_prompt=_get_env("OLLAMA_WARMUP_PROMPT",
            "Ciao, come posso aiutarti?"),
        ollama_keepalive_minutes=_get_env("OLLAMA_KEEPALIVE_MINUTES", 5, int),

        # RAG
        rag_enabled=_get_env("RAG_ENABLED", False, bool),
        rag_docs_path=_get_env("RAG_DOCS_PATH", "data/knowledge"),
        rag_embedding_model=_get_env("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        rag_chunk_size=_get_env("RAG_CHUNK_SIZE", 512, int),
        rag_chunk_overlap=_get_env("RAG_CHUNK_OVERLAP", 50, int),
        rag_top_k=_get_env("RAG_TOP_K", 3, int),

        # Feature flags
        use_http_handlers=_get_env("USE_HTTP_HANDLERS", True, bool),
        memory_enabled=_get_env("MEMORY_ENABLED",
            memory_json.get("enabled", True), bool),
        audit_enabled=_get_env("AUDIT_ENABLED", True, bool),

        # Paths
        data_dir=_get_env("DATA_DIR", "data"),
        logs_dir=_get_env("LOGS_DIR", "logs"),
        config_dir=_get_env("CONFIG_DIR", config_dir),

        # External API
        external_api_key=_get_env("EXTERNAL_API_KEY", ""),
        external_provider=_get_env("EXTERNAL_PROVIDER", "anthropic"),
        external_model=_get_env("EXTERNAL_MODEL", "claude-3-haiku-20240307"),
    )

    return config


# Global config instance (lazy loaded)
_config: Optional[ServiceConfig] = None


def get_config() -> ServiceConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config(env_path: Optional[str] = None) -> ServiceConfig:
    """Reload configuration from files."""
    global _config
    _config = load_config(env_path)
    return _config
