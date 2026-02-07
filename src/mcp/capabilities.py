"""
MCP Capabilities Framework for Antonio Evo (v3.1).

Per Antonio Evo Unified Spec (v3.1):
MCP is a CAPABILITY PROVIDER, not a tool layer.

RULES:
- Antonio NEVER calls MCP directly
- Antonio NEVER selects capabilities
- Antonio NEVER triggers side effects

Antonio MAY ONLY:
- Describe what a capability does
- Help compile task input
- Explain consequences

ALL MCP actions require:
- Explicit approval
- Policy validation
- Audit logging
"""

import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class CapabilityType(Enum):
    """Types of MCP capabilities."""
    READ = "read"          # Read-only operations
    WRITE = "write"        # Write/modify operations
    EXECUTE = "execute"    # Code/command execution
    NETWORK = "network"    # Network operations
    SYSTEM = "system"      # System-level operations


class ApprovalStatus(Enum):
    """Status of capability approval."""
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"


@dataclass
class CapabilityDefinition:
    """
    Definition of an MCP capability.

    Per spec: Antonio describes capabilities, never calls them.
    """
    id: str
    name: str
    description: str
    capability_type: CapabilityType
    side_effects: List[str]  # What this capability can affect
    data_exposure: List[str]  # What data might be exposed
    requires_consent: bool = True
    risk_level: str = "medium"  # low, medium, high, critical

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "type": self.capability_type.value,
            "side_effects": self.side_effects,
            "data_exposure": self.data_exposure,
            "requires_consent": self.requires_consent,
            "risk_level": self.risk_level,
        }

    def get_disclosure(self) -> str:
        """
        Get disclosure text for this capability.

        Per spec: User must always know what can happen.
        """
        lines = [
            f"Capability: {self.name}",
            f"Description: {self.description}",
            f"Risk Level: {self.risk_level.upper()}",
        ]

        if self.side_effects:
            lines.append(f"Side Effects: {', '.join(self.side_effects)}")

        if self.data_exposure:
            lines.append(f"Data Exposure: {', '.join(self.data_exposure)}")

        if self.requires_consent:
            lines.append("Requires: Explicit user consent")

        return "\n".join(lines)


@dataclass
class CapabilityRequest:
    """
    A request to use an MCP capability.

    Per spec: All requests must be explicitly approved.
    """
    id: str
    capability_id: str
    input_data: Dict[str, Any]
    reason: str  # Why this capability is needed
    requested_at: float = field(default_factory=time.time)
    status: ApprovalStatus = ApprovalStatus.PENDING
    approved_by: Optional[str] = None
    approved_at: Optional[float] = None
    expires_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "capability_id": self.capability_id,
            "input_data": self.input_data,
            "reason": self.reason,
            "requested_at": self.requested_at,
            "status": self.status.value,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "expires_at": self.expires_at,
        }


@dataclass
class CapabilityResult:
    """
    Result of a capability execution.

    Per spec: All results are logged for audit.
    """
    request_id: str
    capability_id: str
    success: bool
    output: Any
    error: Optional[str] = None
    executed_at: float = field(default_factory=time.time)
    execution_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "capability_id": self.capability_id,
            "success": self.success,
            "output": self.output if self.success else None,
            "error": self.error,
            "executed_at": self.executed_at,
            "execution_ms": self.execution_ms,
        }


class CapabilityRegistry:
    """
    Registry of available MCP capabilities.

    Per spec: Capabilities are registered, never dynamically invoked.
    """

    def __init__(self):
        """Initialize the capability registry."""
        self._capabilities: Dict[str, CapabilityDefinition] = {}
        self._handlers: Dict[str, Callable] = {}
        self._register_default_capabilities()

    def _register_default_capabilities(self):
        """Register default capabilities with full disclosure."""
        # File read capability
        self.register(CapabilityDefinition(
            id="file_read",
            name="Read File",
            description="Read contents of a specified file from the local filesystem",
            capability_type=CapabilityType.READ,
            side_effects=[],
            data_exposure=["File contents", "File metadata"],
            requires_consent=True,
            risk_level="medium",
        ))

        # File write capability
        self.register(CapabilityDefinition(
            id="file_write",
            name="Write File",
            description="Write or modify contents of a file on the local filesystem",
            capability_type=CapabilityType.WRITE,
            side_effects=["File modification", "Disk space usage"],
            data_exposure=["Written data logged"],
            requires_consent=True,
            risk_level="high",
        ))

        # HTTP fetch capability
        self.register(CapabilityDefinition(
            id="http_fetch",
            name="HTTP Fetch",
            description="Fetch content from a URL over the network",
            capability_type=CapabilityType.NETWORK,
            side_effects=["Network request", "External service contact"],
            data_exposure=["Request URL", "Request headers"],
            requires_consent=True,
            risk_level="medium",
        ))

        # External LLM capability
        self.register(CapabilityDefinition(
            id="external_llm",
            name="External LLM API",
            description="Send query to external LLM service (Claude, GPT)",
            capability_type=CapabilityType.NETWORK,
            side_effects=["API call", "Cost incurred", "Data transmission"],
            data_exposure=["Query text sent to external service"],
            requires_consent=True,
            risk_level="high",
        ))

    def register(
        self,
        capability: CapabilityDefinition,
        handler: Optional[Callable] = None,
    ):
        """Register a capability."""
        self._capabilities[capability.id] = capability
        if handler:
            self._handlers[capability.id] = handler
        logger.debug(f"Registered capability: {capability.id}")

    def get(self, capability_id: str) -> Optional[CapabilityDefinition]:
        """Get a capability definition."""
        return self._capabilities.get(capability_id)

    def list_all(self) -> List[CapabilityDefinition]:
        """List all registered capabilities."""
        return list(self._capabilities.values())

    def describe_capability(self, capability_id: str) -> str:
        """
        Get a description of a capability.

        Per spec: Antonio can only DESCRIBE capabilities.
        """
        cap = self._capabilities.get(capability_id)
        if not cap:
            return f"Unknown capability: {capability_id}"
        return cap.get_disclosure()

    def describe_all(self) -> str:
        """Describe all available capabilities."""
        lines = ["Available Capabilities:", "=" * 40]
        for cap in self._capabilities.values():
            lines.append("")
            lines.append(cap.get_disclosure())
            lines.append("-" * 40)
        return "\n".join(lines)


class CapabilityGate:
    """
    Gate for capability requests.

    Per spec: ALL capability usage requires explicit approval.
    """

    def __init__(self, registry: CapabilityRegistry):
        """Initialize the capability gate."""
        self.registry = registry
        self._pending_requests: Dict[str, CapabilityRequest] = {}
        self._audit_log: List[Dict[str, Any]] = []

    def create_request(
        self,
        capability_id: str,
        input_data: Dict[str, Any],
        reason: str,
    ) -> CapabilityRequest:
        """
        Create a capability request.

        Per spec: Antonio cannot execute, only propose.
        This creates a pending request that requires user approval.
        """
        import uuid

        capability = self.registry.get(capability_id)
        if not capability:
            raise ValueError(f"Unknown capability: {capability_id}")

        request = CapabilityRequest(
            id=str(uuid.uuid4())[:12],
            capability_id=capability_id,
            input_data=input_data,
            reason=reason,
        )

        self._pending_requests[request.id] = request
        self._log_audit("request_created", request.to_dict())

        logger.info(f"Created capability request: {request.id} for {capability_id}")
        return request

    def get_request_disclosure(self, request_id: str) -> str:
        """
        Get disclosure for a capability request.

        Per spec: User must see full disclosure before approving.
        """
        request = self._pending_requests.get(request_id)
        if not request:
            return "Request not found"

        capability = self.registry.get(request.capability_id)
        if not capability:
            return "Capability not found"

        lines = [
            "=" * 50,
            "CAPABILITY REQUEST - APPROVAL NEEDED",
            "=" * 50,
            "",
            capability.get_disclosure(),
            "",
            f"Reason for request: {request.reason}",
            "",
            "Input data:",
            json.dumps(request.input_data, indent=2),
            "",
            "=" * 50,
            "This action requires your explicit approval.",
            "=" * 50,
        ]
        return "\n".join(lines)

    def approve_request(
        self,
        request_id: str,
        approver: str = "user",
        validity_seconds: int = 3600,
    ) -> bool:
        """
        Approve a capability request.

        Per spec: Consent may be per-action, session-scoped, or pre-authorized.
        """
        request = self._pending_requests.get(request_id)
        if not request:
            return False

        request.status = ApprovalStatus.APPROVED
        request.approved_by = approver
        request.approved_at = time.time()
        request.expires_at = time.time() + validity_seconds

        self._log_audit("request_approved", {
            "request_id": request_id,
            "approved_by": approver,
            "expires_at": request.expires_at,
        })

        logger.info(f"Approved capability request: {request_id}")
        return True

    def deny_request(self, request_id: str, reason: str = "") -> bool:
        """Deny a capability request."""
        request = self._pending_requests.get(request_id)
        if not request:
            return False

        request.status = ApprovalStatus.DENIED

        self._log_audit("request_denied", {
            "request_id": request_id,
            "reason": reason,
        })

        logger.info(f"Denied capability request: {request_id}")
        return True

    def is_approved(self, request_id: str) -> bool:
        """Check if a request is approved and not expired."""
        request = self._pending_requests.get(request_id)
        if not request:
            return False

        if request.status != ApprovalStatus.APPROVED:
            return False

        if request.expires_at and time.time() > request.expires_at:
            request.status = ApprovalStatus.EXPIRED
            return False

        return True

    def get_pending_requests(self) -> List[CapabilityRequest]:
        """Get all pending requests."""
        return [
            r for r in self._pending_requests.values()
            if r.status == ApprovalStatus.PENDING
        ]

    def _log_audit(self, action: str, details: Dict[str, Any]):
        """Log to audit trail."""
        entry = {
            "timestamp": time.time(),
            "action": action,
            "details": details,
        }
        self._audit_log.append(entry)

    def get_audit_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent audit log entries."""
        return self._audit_log[-limit:]


class MCPCapabilityProvider:
    """
    MCP Capability Provider for Antonio Evo.

    Per Antonio Evo Unified Spec (v3.1):
    - Provides capabilities, does not execute autonomously
    - All actions gated through explicit approval
    - Full audit trail maintained

    Antonio uses this to:
    1. Describe what capabilities are available
    2. Help compile task input
    3. Explain consequences of actions

    Antonio NEVER uses this to:
    1. Directly call capabilities
    2. Bypass approval
    3. Trigger side effects
    """

    def __init__(self):
        """Initialize the MCP capability provider."""
        self.registry = CapabilityRegistry()
        self.gate = CapabilityGate(self.registry)

    def describe_capabilities(self) -> str:
        """
        Describe all available capabilities.

        Per spec: Antonio can only describe, not invoke.
        """
        return self.registry.describe_all()

    def describe_capability(self, capability_id: str) -> str:
        """Describe a specific capability."""
        return self.registry.describe_capability(capability_id)

    def propose_capability_use(
        self,
        capability_id: str,
        input_data: Dict[str, Any],
        reason: str,
    ) -> Dict[str, Any]:
        """
        Propose using a capability.

        Per spec: Antonio proposes, user approves, code executes.
        Returns a proposal that requires user approval.
        """
        capability = self.registry.get(capability_id)
        if not capability:
            return {
                "success": False,
                "error": f"Unknown capability: {capability_id}",
            }

        # Create the request
        request = self.gate.create_request(capability_id, input_data, reason)

        return {
            "success": True,
            "request_id": request.id,
            "status": "pending_approval",
            "disclosure": self.gate.get_request_disclosure(request.id),
            "message": "This capability request requires your explicit approval.",
        }

    def get_pending_approvals(self) -> List[Dict[str, Any]]:
        """Get all pending approval requests."""
        return [r.to_dict() for r in self.gate.get_pending_requests()]

    def approve(self, request_id: str) -> Dict[str, Any]:
        """Approve a capability request."""
        if self.gate.approve_request(request_id):
            return {
                "success": True,
                "message": f"Request {request_id} approved",
            }
        return {
            "success": False,
            "error": "Request not found or already processed",
        }

    def deny(self, request_id: str, reason: str = "") -> Dict[str, Any]:
        """Deny a capability request."""
        if self.gate.deny_request(request_id, reason):
            return {
                "success": True,
                "message": f"Request {request_id} denied",
            }
        return {
            "success": False,
            "error": "Request not found or already processed",
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get capability provider statistics."""
        return {
            "registered_capabilities": len(self.registry.list_all()),
            "pending_requests": len(self.gate.get_pending_requests()),
            "audit_entries": len(self.gate.get_audit_log(limit=1000)),
        }
