"""
Audit Logger - Persistent logging of all decisions with cryptographic hashing.

Every request, decision, and response is logged with a cryptographic hash chain.
This ensures tamper detection - any modification to historical entries breaks the chain.

Design principles:
- Append-only (never modify existing entries)
- Hash-chained (each entry references previous)
- Fail-safe (errors don't crash the system)
- Deterministic hashing
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class VerificationFailureReason(Enum):
    """Reasons for hash chain verification failure."""
    HASH_MISMATCH = "hash_mismatch"
    PREVIOUS_HASH_MISMATCH = "previous_hash_mismatch"
    SEQUENCE_ERROR = "sequence_error"


@dataclass
class VerificationResult:
    """
    Result of hash chain verification.

    Attributes:
        valid: True if entire chain is valid, False if tampered
        index: Index of first invalid entry (None if valid)
        reason: Reason for failure (None if valid)
        entries_checked: Number of entries verified
    """
    valid: bool
    index: Optional[int] = None
    reason: Optional[str] = None
    entries_checked: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "valid": self.valid,
            "index": self.index,
            "reason": self.reason,
            "entries_checked": self.entries_checked,
        }

from ..models.request import Request
from ..models.policy import PolicyDecision, Classification


# Genesis hash for the first entry in the chain
# This is a well-known constant that marks the beginning of a new chain
GENESIS_HASH = "0" * 64  # 64 zeros (SHA-256 produces 64 hex chars)


@dataclass
class AuditEntry:
    """
    A single audit log entry with hash chain support.

    Attributes:
        timestamp: Unix timestamp of the entry
        timestamp_iso: ISO 8601 formatted timestamp
        event_type: Type of event being logged
        payload: Event-specific data
        previous_hash: Hash of the previous entry (or genesis hash)
        entry_hash: SHA-256 hash of this entry
        sequence: Sequence number in the chain
    """
    timestamp: float
    timestamp_iso: str
    event_type: str
    payload: Dict[str, Any]
    previous_hash: str
    entry_hash: str = ""
    sequence: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "timestamp_iso": self.timestamp_iso,
            "event_type": self.event_type,
            "payload": self.payload,
            "previous_hash": self.previous_hash,
            "entry_hash": self.entry_hash,
            "sequence": self.sequence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditEntry":
        """Create AuditEntry from dictionary."""
        return cls(
            timestamp=data.get("timestamp", 0),
            timestamp_iso=data.get("timestamp_iso", ""),
            event_type=data.get("event_type", ""),
            payload=data.get("payload", {}),
            previous_hash=data.get("previous_hash", GENESIS_HASH),
            entry_hash=data.get("entry_hash", ""),
            sequence=data.get("sequence", 0),
        )


def compute_entry_hash(
    timestamp: float,
    event_type: str,
    payload: Dict[str, Any],
    previous_hash: str,
) -> str:
    """
    Compute SHA-256 hash for an audit entry.

    The hash is computed over a canonical JSON representation of:
    - timestamp
    - event_type
    - payload
    - previous_hash

    Args:
        timestamp: Unix timestamp
        event_type: Type of event
        payload: Event data
        previous_hash: Hash of previous entry

    Returns:
        Hex-encoded SHA-256 hash (64 characters)
    """
    # Create canonical representation for hashing
    # Sort keys to ensure deterministic serialization
    hash_input = {
        "timestamp": timestamp,
        "event_type": event_type,
        "payload": payload,
        "previous_hash": previous_hash,
    }

    # Serialize to JSON with sorted keys for determinism
    canonical = json.dumps(hash_input, sort_keys=True, separators=(",", ":"))

    # Compute SHA-256
    hash_bytes = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    return hash_bytes


class AuditLogger:
    """
    Audit logger with cryptographic hash chain.

    Every entry includes:
    - A SHA-256 hash of the entry content
    - The hash of the previous entry (creating a chain)

    This makes tampering detectable - modifying any entry
    would break the chain from that point forward.

    Usage:
        logger = AuditLogger()
        logger.log(request, classification, decision, response, elapsed_ms)

        # Or log generic events
        logger.log_event("custom_event", {"key": "value"})

        # Get recent entries
        entries = logger.get_recent(10)
    """

    def __init__(self, log_path: str = "logs/audit.jsonl"):
        """
        Initialize audit logger.

        Args:
            log_path: Path to the JSONL log file
        """
        self.log_path = log_path
        self._previous_hash = GENESIS_HASH
        self._sequence = 0
        self._ensure_dir()
        self._load_chain_state()

    def _ensure_dir(self):
        """Ensure log directory exists."""
        log_dir = os.path.dirname(self.log_path)
        if log_dir:
            try:
                os.makedirs(log_dir, exist_ok=True)
            except OSError:
                # Directory creation failed, will handle in write
                pass

    def _load_chain_state(self):
        """
        Load chain state from existing log file.

        Reads the last entry to get the previous hash and sequence number.
        If file doesn't exist or is empty, starts a new chain.
        """
        try:
            if not os.path.exists(self.log_path):
                # New chain
                self._previous_hash = GENESIS_HASH
                self._sequence = 0
                return

            with open(self.log_path, "r", encoding="utf-8") as f:
                last_line = None
                for line in f:
                    if line.strip():
                        last_line = line

                if last_line:
                    last_entry = json.loads(last_line)
                    self._previous_hash = last_entry.get("entry_hash", GENESIS_HASH)
                    self._sequence = last_entry.get("sequence", 0)
                else:
                    # Empty file
                    self._previous_hash = GENESIS_HASH
                    self._sequence = 0

        except (IOError, json.JSONDecodeError, KeyError):
            # If we can't read the file, start fresh
            # This is a fail-safe - we don't want to crash
            self._previous_hash = GENESIS_HASH
            self._sequence = 0

    def log(
        self,
        request: Request,
        classification: Classification,
        decision: PolicyDecision,
        response: Dict[str, Any],
        elapsed_ms: int,
        memory_operation: Optional[Any] = None,
    ) -> Optional[AuditEntry]:
        """
        Log a complete request cycle.

        This is append-only. Never modify existing logs.

        Args:
            request: The normalized request
            classification: Classification result
            decision: Policy decision
            response: Handler response
            elapsed_ms: Processing time in milliseconds
            memory_operation: EvoMemory operation details (optional)

        Returns:
            The created AuditEntry, or None if logging failed
        """
        payload = {
            # Request info (truncated for privacy)
            "request": {
                "id": request.request_id,
                "text_preview": request.text[:100] if request.text else "",
                "text_length": len(request.text) if request.text else 0,
                "modality": request.modality.value,
                "task_type": request.task_type,
                "source": request.source,
                "session_id": request.session_id,
            },

            # Classification
            "classification": classification.to_dict(),

            # Policy decision
            "decision": decision.to_dict(),

            # Response summary
            "response": {
                "success": response.get("success", False),
                "error": response.get("error"),
                "error_code": response.get("error_code"),
                "used_external": response.get("_meta", {}).get("used_external", False),
                "neuron_stored": response.get("_meta", {}).get("neuron_stored", False),
                "neuron_id": response.get("_meta", {}).get("neuron_id"),
            },

            # Performance
            "elapsed_ms": elapsed_ms,
        }

        # Add memory operation if present
        if memory_operation is not None:
            try:
                payload["memory_operation"] = memory_operation.to_dict()
            except AttributeError:
                # memory_operation doesn't have to_dict, skip it
                pass

        return self.log_event("request_cycle", payload)

    def log_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
    ) -> Optional[AuditEntry]:
        """
        Log a generic event with hash chaining.

        Args:
            event_type: Type of event (e.g., "request_cycle", "system_start")
            payload: Event-specific data

        Returns:
            The created AuditEntry, or None if logging failed
        """
        timestamp = time.time()
        timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(timestamp))

        # Compute hash for this entry
        entry_hash = compute_entry_hash(
            timestamp=timestamp,
            event_type=event_type,
            payload=payload,
            previous_hash=self._previous_hash,
        )

        # Create entry
        self._sequence += 1
        entry = AuditEntry(
            timestamp=timestamp,
            timestamp_iso=timestamp_iso,
            event_type=event_type,
            payload=payload,
            previous_hash=self._previous_hash,
            entry_hash=entry_hash,
            sequence=self._sequence,
        )

        # Write to file
        if self._write(entry):
            # Update chain state only on successful write
            self._previous_hash = entry_hash
            return entry
        else:
            # Roll back sequence on failure
            self._sequence -= 1
            return None

    def _write(self, entry: AuditEntry) -> bool:
        """
        Write entry to log file.

        Args:
            entry: The audit entry to write

        Returns:
            True if write succeeded, False otherwise
        """
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
            return True
        except IOError as e:
            # Log errors should not crash the system
            # Print to stderr as a last resort
            import sys
            print(f"[AUDIT WARNING] Failed to write log: {e}", file=sys.stderr)
            return False

    def get_recent(self, n: int = 10) -> List[AuditEntry]:
        """
        Get last N log entries.

        Args:
            n: Number of entries to retrieve

        Returns:
            List of AuditEntry objects (newest last)
        """
        try:
            if not os.path.exists(self.log_path):
                return []

            with open(self.log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                entries = []
                for line in lines[-n:]:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            entries.append(AuditEntry.from_dict(data))
                        except json.JSONDecodeError:
                            continue
                return entries

        except IOError:
            return []

    def get_all(self) -> List[AuditEntry]:
        """
        Get all log entries.

        Warning: May be slow for large log files.

        Returns:
            List of all AuditEntry objects
        """
        try:
            if not os.path.exists(self.log_path):
                return []

            entries = []
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            entries.append(AuditEntry.from_dict(data))
                        except json.JSONDecodeError:
                            continue
            return entries

        except IOError:
            return []

    def get_entry_by_sequence(self, sequence: int) -> Optional[AuditEntry]:
        """
        Get entry by sequence number.

        Args:
            sequence: Sequence number to find

        Returns:
            AuditEntry if found, None otherwise
        """
        try:
            if not os.path.exists(self.log_path):
                return None

            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if data.get("sequence") == sequence:
                                return AuditEntry.from_dict(data)
                        except json.JSONDecodeError:
                            continue
            return None

        except IOError:
            return None

    @property
    def current_hash(self) -> str:
        """Get the current chain head hash."""
        return self._previous_hash

    @property
    def current_sequence(self) -> int:
        """Get the current sequence number."""
        return self._sequence

    def is_genesis(self) -> bool:
        """Check if this is a new/empty chain."""
        return self._previous_hash == GENESIS_HASH and self._sequence == 0

    def verify_chain(self) -> VerificationResult:
        """
        Verify the integrity of the entire hash chain.

        Reads all entries and verifies:
        1. Each entry's hash matches recomputed hash
        2. Each entry's previous_hash matches prior entry's entry_hash
        3. Sequence numbers are consecutive starting from 1

        Stops at the FIRST detected inconsistency.

        Returns:
            VerificationResult with valid=True if chain is intact,
            or valid=False with index and reason of first failure.

        Note:
            This method never raises exceptions. All errors result in
            a VerificationResult with appropriate failure information.
        """
        try:
            # Handle nonexistent file
            if not os.path.exists(self.log_path):
                return VerificationResult(valid=True, entries_checked=0)

            entries = []
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            entries.append(data)
                        except json.JSONDecodeError:
                            # Corrupted entry - cannot verify
                            return VerificationResult(
                                valid=False,
                                index=len(entries),
                                reason=VerificationFailureReason.HASH_MISMATCH.value,
                                entries_checked=len(entries),
                            )

            # Empty log is valid
            if not entries:
                return VerificationResult(valid=True, entries_checked=0)

            # Verify each entry
            expected_previous_hash = GENESIS_HASH
            expected_sequence = 1

            for i, entry_data in enumerate(entries):
                # Check sequence
                actual_sequence = entry_data.get("sequence", 0)
                if actual_sequence != expected_sequence:
                    return VerificationResult(
                        valid=False,
                        index=i,
                        reason=VerificationFailureReason.SEQUENCE_ERROR.value,
                        entries_checked=i,
                    )

                # Check previous_hash linkage
                actual_previous_hash = entry_data.get("previous_hash", "")
                if actual_previous_hash != expected_previous_hash:
                    return VerificationResult(
                        valid=False,
                        index=i,
                        reason=VerificationFailureReason.PREVIOUS_HASH_MISMATCH.value,
                        entries_checked=i,
                    )

                # Recompute hash and compare
                computed_hash = compute_entry_hash(
                    timestamp=entry_data.get("timestamp", 0),
                    event_type=entry_data.get("event_type", ""),
                    payload=entry_data.get("payload", {}),
                    previous_hash=actual_previous_hash,
                )

                actual_hash = entry_data.get("entry_hash", "")
                if computed_hash != actual_hash:
                    return VerificationResult(
                        valid=False,
                        index=i,
                        reason=VerificationFailureReason.HASH_MISMATCH.value,
                        entries_checked=i,
                    )

                # Update expectations for next entry
                expected_previous_hash = actual_hash
                expected_sequence += 1

            # All entries verified
            return VerificationResult(valid=True, entries_checked=len(entries))

        except IOError:
            # File read error - cannot verify
            return VerificationResult(
                valid=False,
                index=0,
                reason=VerificationFailureReason.HASH_MISMATCH.value,
                entries_checked=0,
            )
