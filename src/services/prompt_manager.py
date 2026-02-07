"""
System Prompt Manager for Antonio Evo.

Dynamically builds system prompts based on:
- Runtime profile and constraints
- Personality traits
- Emotional context
- Digital twin style
- Available capabilities
- Memory context

Philosophy: Context is everything. Prompts adapt to reality.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PromptContext:
    """Context for prompt generation."""
    # Runtime
    profile_name: str = "evo-standard"
    max_tokens: int = 1024
    verbosity: str = "standard"  # minimal, standard, detailed

    # Personality (0-100 scales)
    humor: int = 50
    formality: int = 50
    verbosity_trait: int = 50
    empathy: int = 50
    curiosity: int = 50
    patience: int = 50
    creativity: int = 50

    # Emotional
    user_emotion: str = "neutral"
    tone_recommendation: str = "balanced"
    emotional_trend: str = "stable"

    # Digital Twin
    twin_ready: bool = False
    twin_style_prompt: Optional[str] = None

    # Capabilities
    has_memory: bool = True
    has_rag: bool = False
    has_image_gen: bool = False
    has_image_analysis: bool = False
    has_external_llm: bool = False

    # Memory
    relevant_memories: List[str] = field(default_factory=list)
    memory_context_summary: Optional[str] = None

    # Session
    session_id: Optional[str] = None
    turn_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "max_tokens": self.max_tokens,
            "verbosity": self.verbosity,
            "personality": {
                "humor": self.humor,
                "formality": self.formality,
                "empathy": self.empathy,
                "curiosity": self.curiosity,
                "patience": self.patience,
                "creativity": self.creativity,
            },
            "emotional": {
                "user_emotion": self.user_emotion,
                "tone_recommendation": self.tone_recommendation,
                "emotional_trend": self.emotional_trend,
            },
            "twin_ready": self.twin_ready,
            "capabilities": {
                "memory": self.has_memory,
                "rag": self.has_rag,
                "image_gen": self.has_image_gen,
                "image_analysis": self.has_image_analysis,
                "external_llm": self.has_external_llm,
            },
            "session": {
                "id": self.session_id,
                "turn_count": self.turn_count,
            },
        }


class SystemPromptManager:
    """
    Builds dynamic system prompts for Antonio.

    Sections:
    1. Core Identity
    2. Runtime Constraints
    3. Personality Guidelines
    4. Emotional Adaptation
    5. Memory Context
    6. Capability Awareness
    7. Response Format
    """

    # Core identity template
    CORE_IDENTITY = """# Antonio Evo - AI Assistant

You are Antonio, a local-first AI assistant with evolutionary memory.

Core Principles:
- Be helpful, accurate, and honest
- Respect privacy - all data stays local
- Learn and adapt from interactions
- Never pretend to have capabilities you don't have
"""

    # Verbosity templates
    VERBOSITY_TEMPLATES = {
        "minimal": "Keep responses brief and focused. Use 1-2 sentences when possible.",
        "standard": "Provide clear, well-structured responses. Balance detail with conciseness.",
        "detailed": "Give comprehensive responses with examples and explanations when helpful.",
    }

    def __init__(self, orchestrator: Optional[Any] = None):
        """
        Initialize prompt manager.

        Args:
            orchestrator: Reference to main Orchestrator
        """
        self.orchestrator = orchestrator
        self._cache: Dict[str, str] = {}

    def set_orchestrator(self, orchestrator: Any) -> None:
        """Set orchestrator reference."""
        self.orchestrator = orchestrator

    def build_context(self) -> PromptContext:
        """Build context from current system state."""
        ctx = PromptContext()

        if not self.orchestrator:
            return ctx

        # Runtime profile
        if hasattr(self.orchestrator, 'profile_manager'):
            pm = self.orchestrator.profile_manager
            ctx.profile_name = pm.get_active_profile().value
            constraints = pm.get_response_constraints()
            ctx.max_tokens = constraints.get("max_tokens", 1024)
            ctx.verbosity = constraints.get("verbosity", "standard")

            # Capabilities from profile
            caps = pm.get_capabilities()
            ctx.has_memory = caps.memory_enabled
            ctx.has_rag = caps.rag_enabled
            ctx.has_image_gen = caps.image_generation
            ctx.has_image_analysis = caps.image_analysis
            ctx.has_external_llm = caps.external_llm_fallback

        # Personality traits
        if hasattr(self.orchestrator, 'personality_engine') and self.orchestrator.personality_engine:
            profile = self.orchestrator.personality_engine.get_profile()
            traits = profile.to_dict().get("traits", {})
            ctx.humor = traits.get("humor", 50)
            ctx.formality = traits.get("formality", 50)
            ctx.verbosity_trait = traits.get("verbosity", 50)
            ctx.empathy = traits.get("empathy", 50)
            ctx.curiosity = traits.get("curiosity", 50)
            ctx.patience = traits.get("patience", 50)
            ctx.creativity = traits.get("creativity", 50)

        # Digital Twin
        if hasattr(self.orchestrator, 'digital_twin') and self.orchestrator.digital_twin:
            ctx.twin_ready = self.orchestrator.digital_twin.is_ready()
            if ctx.twin_ready:
                ctx.twin_style_prompt = self.orchestrator.digital_twin.generate_style_prompt()

        # Session info
        ctx.session_id = getattr(self.orchestrator, 'current_session_id', None)

        return ctx

    def build_prompt(
        self,
        context: Optional[PromptContext] = None,
        emotional_context: Optional[Any] = None,
        memory_context: Optional[Any] = None,
        include_sections: Optional[List[str]] = None,
    ) -> str:
        """
        Build complete system prompt.

        Args:
            context: Optional pre-built context
            emotional_context: Emotional analysis result
            memory_context: Memory retrieval result
            include_sections: Specific sections to include (default: all)

        Returns:
            Complete system prompt string
        """
        ctx = context or self.build_context()

        # Update context with emotional info
        if emotional_context:
            ctx.user_emotion = getattr(emotional_context, 'current_state', 'neutral')
            if hasattr(ctx.user_emotion, 'value'):
                ctx.user_emotion = ctx.user_emotion.value
            ctx.tone_recommendation = getattr(emotional_context, 'tone_recommendation', 'balanced')
            if hasattr(ctx.tone_recommendation, 'value'):
                ctx.tone_recommendation = ctx.tone_recommendation.value
            ctx.emotional_trend = getattr(emotional_context, 'emotional_trend', 'stable')

        # Update context with memory info
        if memory_context and hasattr(memory_context, 'has_relevant_memory') and memory_context.has_relevant_memory:
            ctx.memory_context_summary = self._summarize_memory(memory_context)

        # Build sections
        sections = []

        all_sections = include_sections or [
            "identity",
            "constraints",
            "personality",
            "emotional",
            "memory",
            "capabilities",
            "format",
        ]

        if "identity" in all_sections:
            sections.append(self._build_identity_section(ctx))

        if "constraints" in all_sections:
            sections.append(self._build_constraints_section(ctx))

        if "personality" in all_sections:
            sections.append(self._build_personality_section(ctx))

        if "emotional" in all_sections and ctx.user_emotion != "neutral":
            sections.append(self._build_emotional_section(ctx))

        if "memory" in all_sections and ctx.memory_context_summary:
            sections.append(self._build_memory_section(ctx))

        if "capabilities" in all_sections:
            sections.append(self._build_capabilities_section(ctx))

        if "format" in all_sections:
            sections.append(self._build_format_section(ctx))

        return "\n\n".join(filter(None, sections))

    def _build_identity_section(self, ctx: PromptContext) -> str:
        """Build core identity section."""
        identity = self.CORE_IDENTITY.strip()

        # Add Digital Twin style if ready
        if ctx.twin_ready and ctx.twin_style_prompt:
            identity += f"\n\n## Communication Style\n{ctx.twin_style_prompt}"

        return identity

    def _build_constraints_section(self, ctx: PromptContext) -> str:
        """Build runtime constraints section."""
        lines = ["## Runtime Constraints"]
        lines.append(f"- Profile: {ctx.profile_name}")
        lines.append(f"- Response limit: {ctx.max_tokens} tokens")
        lines.append(f"- Verbosity: {ctx.verbosity}")

        return "\n".join(lines)

    def _build_personality_section(self, ctx: PromptContext) -> str:
        """Build personality guidelines section."""
        lines = ["## Personality Guidelines"]

        # Humor
        if ctx.humor < 30:
            lines.append("- Keep responses professional and serious")
        elif ctx.humor > 70:
            lines.append("- Feel free to use humor and wit when appropriate")

        # Formality
        if ctx.formality < 30:
            lines.append("- Use casual, conversational language")
        elif ctx.formality > 70:
            lines.append("- Maintain formal, professional tone")

        # Empathy
        if ctx.empathy > 60:
            lines.append("- Show understanding and emotional awareness")

        # Curiosity
        if ctx.curiosity > 60:
            lines.append("- Ask follow-up questions when topics are interesting")

        # Patience
        if ctx.patience > 60:
            lines.append("- Take time to explain things thoroughly")

        # Creativity
        if ctx.creativity > 60:
            lines.append("- Offer creative suggestions and alternatives")

        if len(lines) == 1:
            return ""  # No specific personality adjustments

        return "\n".join(lines)

    def _build_emotional_section(self, ctx: PromptContext) -> str:
        """Build emotional adaptation section."""
        lines = ["## Emotional Context"]
        lines.append(f"User appears: {ctx.user_emotion}")
        lines.append(f"Trend: {ctx.emotional_trend}")

        # Tone recommendations
        tone_guides = {
            "supportive": "Be encouraging and validating",
            "patient": "Take extra time to explain, don't rush",
            "enthusiastic": "Match the user's energy and excitement",
            "calming": "Use a soothing, reassuring tone",
            "concise": "Keep responses brief and to the point",
            "empathetic": "Acknowledge feelings before addressing content",
            "balanced": "Maintain a balanced, neutral tone",
        }

        guide = tone_guides.get(ctx.tone_recommendation, "")
        if guide:
            lines.append(f"Tone: {guide}")

        return "\n".join(lines)

    def _build_memory_section(self, ctx: PromptContext) -> str:
        """Build memory context section."""
        if not ctx.memory_context_summary:
            return ""

        lines = ["## Relevant Memory"]
        lines.append(ctx.memory_context_summary)

        return "\n".join(lines)

    def _build_capabilities_section(self, ctx: PromptContext) -> str:
        """Build capabilities awareness section."""
        lines = ["## Available Capabilities"]

        if ctx.has_memory:
            lines.append("- Memory: Can recall past interactions")
        if ctx.has_rag:
            lines.append("- Documents: Can search knowledge base")
        if ctx.has_image_gen:
            lines.append("- Image generation: Can create images from descriptions")
        if ctx.has_image_analysis:
            lines.append("- Image analysis: Can understand image content")
        if ctx.has_external_llm:
            lines.append("- External AI: Can use cloud AI for complex tasks")

        # What we cannot do
        cant_do = []
        if not ctx.has_image_gen:
            cant_do.append("generate images")
        if not ctx.has_image_analysis:
            cant_do.append("analyze images")
        if not ctx.has_rag:
            cant_do.append("search documents")

        if cant_do:
            lines.append(f"\nNote: Cannot {', '.join(cant_do)} in current profile.")

        return "\n".join(lines)

    def _build_format_section(self, ctx: PromptContext) -> str:
        """Build response format section."""
        lines = ["## Response Format"]

        verbosity_guide = self.VERBOSITY_TEMPLATES.get(ctx.verbosity, "")
        if verbosity_guide:
            lines.append(verbosity_guide)

        # Add general guidelines
        lines.append("\nGuidelines:")
        lines.append("- Be direct and clear")
        lines.append("- Use markdown for structure when helpful")
        lines.append("- Acknowledge uncertainty rather than guessing")
        lines.append("- Ask for clarification if the request is ambiguous")

        return "\n".join(lines)

    def _summarize_memory(self, memory_context: Any) -> str:
        """Summarize memory context for prompt."""
        if not memory_context or not hasattr(memory_context, 'relevant_neurons'):
            return ""

        neurons = memory_context.relevant_neurons[:3]  # Top 3
        if not neurons:
            return ""

        summaries = []
        for rn in neurons:
            if hasattr(rn, 'neuron') and hasattr(rn.neuron, 'input'):
                summary = rn.neuron.input[:100]
                if len(rn.neuron.input) > 100:
                    summary += "..."
                summaries.append(f"- {summary}")

        if summaries:
            return "Past relevant interactions:\n" + "\n".join(summaries)
        return ""

    def get_stats(self) -> Dict[str, Any]:
        """Get prompt manager statistics."""
        ctx = self.build_context()
        return {
            "version": "1.0",
            "enabled": True,
            "current_context": ctx.to_dict(),
        }


# Singleton instance
_prompt_manager: Optional[SystemPromptManager] = None


def get_prompt_manager(orchestrator: Optional[Any] = None) -> SystemPromptManager:
    """Get or create the prompt manager singleton."""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = SystemPromptManager(orchestrator)
    elif orchestrator and not _prompt_manager.orchestrator:
        _prompt_manager.set_orchestrator(orchestrator)
    return _prompt_manager
