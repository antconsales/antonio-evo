"""
Neuron Creator for EvoMemory.

Creates and stores neurons from successful interactions.
Called AFTER a response is generated.
"""

from typing import Optional, Dict, Any
import hashlib

from ..memory.neuron import (
    Neuron,
    NeuronCreate,
    Mood,
    Persona,
    MemoryOperation,
)
from ..memory.storage import MemoryStorage
from ..models.request import Request
from ..models.policy import PolicyDecision, Classification


class NeuronCreator:
    """
    Creates neurons from successful interactions.

    Decides:
    - Whether to store an interaction
    - What confidence score to assign
    - What mood to annotate
    """

    def __init__(
        self,
        storage: MemoryStorage,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize creator.

        Args:
            storage: MemoryStorage instance
            config: Learning configuration
        """
        self.storage = storage
        self.config = config or {}

        # Config defaults
        self.auto_store = self.config.get("auto_store", True)
        self.min_confidence_to_store = self.config.get("min_confidence_to_store", 0.4)
        self.base_confidence = self.config.get("base_confidence", 0.5)

    def should_store(
        self,
        request: Request,
        response: Dict[str, Any],
        decision: PolicyDecision,
    ) -> bool:
        """
        Decide if this interaction should be stored as a neuron.

        Args:
            request: The original request
            response: The generated response
            decision: The policy decision

        Returns:
            True if should store, False otherwise
        """
        # Check if storage is disabled
        if not self.auto_store:
            return False

        # Check if policy says not to store
        if not decision.store_as_neuron:
            return False

        # Don't store rejected requests
        if decision.reject_reason is not None:
            return False

        # Don't store failed responses
        if not response.get("success", False):
            return False

        # Don't store empty responses
        output = response.get("output") or response.get("text", "")
        if not output or len(str(output).strip()) < 10:
            return False

        # Don't store very short inputs (likely noise)
        if len(request.text.strip()) < 3:
            return False

        return True

    def calculate_confidence(
        self,
        request: Request,
        response: Dict[str, Any],
        decision: PolicyDecision,
        classification: Classification,
    ) -> float:
        """
        Calculate initial confidence score for a neuron.

        Higher confidence for:
        - Local handler (no external API)
        - High classification confidence
        - Longer, more detailed responses
        - Memory-informed responses

        Args:
            request: The original request
            response: The generated response
            decision: The policy decision
            classification: The classification result

        Returns:
            Confidence score (0.0-1.0)
        """
        confidence = self.base_confidence

        # Boost for local handling (no cloud dependency)
        if not response.get("_meta", {}).get("used_external", False):
            confidence += 0.1

        # Boost based on classification confidence
        confidence += classification.confidence * 0.1

        # Boost for memory-informed responses
        if classification.memory_informed:
            confidence += 0.05

        # Boost for longer responses (indicates thoroughness)
        output = response.get("output") or response.get("text", "")
        output_len = len(str(output))
        if output_len > 500:
            confidence += 0.1
        elif output_len > 200:
            confidence += 0.05

        # Slight penalty for external fallback (less reliable for future local use)
        if decision.allow_external and response.get("_meta", {}).get("used_external"):
            confidence -= 0.1

        # Clamp to valid range
        return max(0.1, min(1.0, confidence))

    def determine_mood(
        self,
        request: Request,
        response: Dict[str, Any],
        decision: PolicyDecision,
    ) -> Mood:
        """
        Determine mood annotation for the neuron.

        Based on persona and response characteristics.

        Args:
            request: The original request
            response: The generated response
            decision: The policy decision

        Returns:
            Mood enum value
        """
        # Base on persona
        if decision.persona == Persona.LOGIC:
            return Mood.ANALYTICAL

        if decision.persona == Persona.SOCIAL:
            # Check for question keywords
            text_lower = request.text.lower()
            if any(kw in text_lower for kw in ["help", "please", "how do", "can you"]):
                return Mood.HELPFUL
            if any(kw in text_lower for kw in ["what is", "why", "explain"]):
                return Mood.CURIOUS
            return Mood.FRIENDLY

        # Default
        return Mood.NEUTRAL

    def create_neuron(
        self,
        request: Request,
        response: Dict[str, Any],
        decision: PolicyDecision,
        classification: Classification,
    ) -> Optional[Neuron]:
        """
        Create and store a neuron from an interaction.

        This is the main entry point, called after response generation.

        Args:
            request: The original request
            response: The generated response
            decision: The policy decision
            classification: The classification result

        Returns:
            Created Neuron if stored, None if skipped
        """
        # Check if we should store
        if not self.should_store(request, response, decision):
            return None

        # Calculate confidence
        confidence = self.calculate_confidence(
            request, response, decision, classification
        )

        # Skip if confidence too low
        if confidence < self.min_confidence_to_store:
            return None

        # Determine mood
        mood = self.determine_mood(request, response, decision)

        # Determine persona
        persona = decision.persona
        if persona == Persona.AUTO:
            # Infer from handler
            if decision.handler.value.endswith("_logic"):
                persona = Persona.LOGIC
            elif decision.handler.value.endswith("_social"):
                persona = Persona.SOCIAL
            else:
                persona = Persona.SOCIAL  # Default

        # Get output text
        output = response.get("output") or response.get("text", "")

        # Extract multimodal attachment summaries (v8.0)
        attachment_summary = None
        if hasattr(request, "attachments") and request.attachments:
            summaries = []
            for att in request.attachments:
                desc = getattr(att, "description", None)
                if desc:
                    summaries.append(desc)
            if summaries:
                attachment_summary = " | ".join(summaries)

        # Create neuron
        neuron_create = NeuronCreate(
            input=request.text,
            output=str(output),
            handler=decision.handler.value,
            persona=persona,
            mood=mood,
            confidence=confidence,
            session_id=request.session_id,
            request_id=request.request_id,
            classification_domain=classification.domain,
            attachment_summary=attachment_summary,
        )

        # Store and return
        neuron = self.storage.store(neuron_create)
        return neuron

    def create_memory_operation(
        self,
        request: Request,
        neuron: Optional[Neuron],
    ) -> MemoryOperation:
        """
        Create a MemoryOperation record for audit logging.

        Args:
            request: The original request
            neuron: The created neuron (or None if not stored)

        Returns:
            MemoryOperation for audit log
        """
        # Get retrieval info from memory context
        retrieved_ids = []
        retrieval_scores = []

        if request.memory_context:
            for rn in request.memory_context.relevant_neurons:
                retrieved_ids.append(rn.neuron.id)
                retrieval_scores.append(rn.relevance_score)

        # Determine operation type
        if neuron and retrieved_ids:
            op_type = "retrieve_and_store"
        elif neuron:
            op_type = "store"
        elif retrieved_ids:
            op_type = "retrieve"
        else:
            op_type = "none"

        return MemoryOperation(
            operation_type=op_type,
            neuron_ids=retrieved_ids,
            retrieval_scores=retrieval_scores,
            query_hash=hashlib.sha256(request.text.encode()).hexdigest()[:16],
            stored_neuron_id=neuron.id if neuron else None,
            confidence_assigned=neuron.confidence if neuron else None,
        )
