"""
Services module - Background services and utilities.

Includes:
- OllamaWarmup: Model preloading and keep-alive for Ollama
- PatternAnalyzer, ProactiveService: Proactive mode (v2.2)
- PersonalityEvolutionEngine: Personality evolution (v2.3)
- DigitalTwin: Digital twin (v3.0)
- LLMManager: Multi-LLM awareness
- TaskManager: Task system with approval workflow
- SystemPromptManager: Dynamic system prompt builder
- DocumentParser: OCR/VLM document parsing (v4.1)
"""

from .ollama_warmup import OllamaWarmup
from .document_parser import DocumentParser, DocumentParseResult
from .proactive import PatternAnalyzer, ProactiveService
from .personality import PersonalityEvolutionEngine, FeedbackSignal, PersonalityTrait
from .digital_twin import DigitalTwin
from .llm_manager import LLMManager, LLMProvider, LLMEndpoint, LLMRequest, LLMResponse, get_llm_manager
from .task_system import TaskManager, Task, TaskSchema, TaskType, TaskStatus, TaskPriority, ApprovalLevel, get_task_manager
from .prompt_manager import SystemPromptManager, PromptContext, get_prompt_manager

__all__ = [
    "OllamaWarmup",
    "PatternAnalyzer",
    "ProactiveService",
    "PersonalityEvolutionEngine",
    "FeedbackSignal",
    "PersonalityTrait",
    "DigitalTwin",
    "LLMManager",
    "LLMProvider",
    "LLMEndpoint",
    "LLMRequest",
    "LLMResponse",
    "get_llm_manager",
    "TaskManager",
    "Task",
    "TaskSchema",
    "TaskType",
    "TaskStatus",
    "TaskPriority",
    "ApprovalLevel",
    "get_task_manager",
    "SystemPromptManager",
    "PromptContext",
    "get_prompt_manager",
    "DocumentParser",
    "DocumentParseResult",
]
