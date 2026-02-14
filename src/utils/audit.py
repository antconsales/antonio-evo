"""
Audit Logger - Persistent logging of all decisions with cryptographic hashing.

Every request, decision, and response is logged with a cryptographic hash chain.
This ensures tamper detection - any modification to historical entries breaks the chain.

v8.5: External digest file (logs/audit_digest.sha256) for independent verification.
The agent writes to this file but never reads it during operation — it exists
solely for external monitoring tools to `tail -f` and verify integrity.

Design principles:
- Append-only (never modify existing entries)
- Hash-chained (each entry references previous)
- External digest for independent verification
- Fail-safe (errors don't crash the system)
- Deterministic hashing
"""

import glob as glob_module
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

    def __init__(
        self,
        log_path: str = "logs/audit.jsonl",
        max_size_mb: int = 50,
        max_backups: int = 10,
        retention_days: int = 30,
    ):
        """
        Initialize audit logger.

        Args:
            log_path: Path to the JSONL log file
            max_size_mb: Maximum log file size before rotation (default 50MB)
            max_backups: Maximum number of backup files to keep (default 10)
            retention_days: Delete backups older than this (default 30 days)
        """
        self.log_path = log_path
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_backups = max_backups
        self.retention_days = retention_days
        self._previous_hash = GENESIS_HASH
        self._sequence = 0
        self._ensure_dir()
        self._enforce_retention()
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

    def _check_rotation(self):
        """
        Check if log file needs rotation based on size.

        When rotated:
        - audit.jsonl.{N} → audit.jsonl.{N+1} (shift existing backups)
        - audit.jsonl → audit.jsonl.1
        - Start new chain with genesis hash
        - Delete backups exceeding max_backups
        """
        try:
            if not os.path.exists(self.log_path):
                return

            file_size = os.path.getsize(self.log_path)
            if file_size < self.max_size_bytes:
                return

            # Shift existing backups: .9 → .10, .8 → .9, etc.
            for i in range(self.max_backups, 0, -1):
                old_path = f"{self.log_path}.{i}"
                new_path = f"{self.log_path}.{i + 1}"
                if os.path.exists(old_path):
                    if i >= self.max_backups:
                        os.remove(old_path)  # Delete oldest beyond limit
                    else:
                        os.rename(old_path, new_path)

            # Rotate current log to .1
            os.rename(self.log_path, f"{self.log_path}.1")

            # Reset chain state for new file
            self._previous_hash = GENESIS_HASH
            self._sequence = 0

        except OSError:
            # Rotation failed, continue writing to current file
            pass

    def _enforce_retention(self):
        """
        Delete backup files older than retention_days.

        Called at initialization to clean up old logs.
        """
        try:
            cutoff_time = time.time() - (self.retention_days * 86400)
            pattern = f"{self.log_path}.*"
            for backup_path in glob_module.glob(pattern):
                try:
                    if os.path.getmtime(backup_path) < cutoff_time:
                        os.remove(backup_path)
                except OSError:
                    continue
        except Exception:
            # Retention enforcement failed, non-critical
            pass

    def _write(self, entry: AuditEntry) -> bool:
        """
        Write entry to log file and external digest.

        Checks for rotation before writing.

        Args:
            entry: The audit entry to write

        Returns:
            True if write succeeded, False otherwise
        """
        try:
            # Check if rotation needed before writing
            self._check_rotation()

            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

            # Write external digest (v8.5)
            self._write_digest(entry)

            return True
        except IOError as e:
            # Log errors should not crash the system
            # Print to stderr as a last resort
            import sys
            print(f"[AUDIT WARNING] Failed to write log: {e}", file=sys.stderr)
            return False

    def _write_digest(self, entry: AuditEntry) -> None:
        """
        Append entry summary to external digest file (v8.5).

        Format: sequence|timestamp_iso|event_type|entry_hash
        Designed for external monitoring: `tail -f logs/audit_digest.sha256`
        The agent writes here but never reads during operation.
        """
        digest_path = os.path.join(os.path.dirname(self.log_path), "audit_digest.sha256")
        try:
            with open(digest_path, "a", encoding="utf-8") as f:
                f.write(f"{entry.sequence}|{entry.timestamp_iso}|{entry.event_type}|{entry.entry_hash}\n")
        except IOError:
            pass  # Non-critical — digest is supplementary

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

    # ---- Governance Audit (v8.5) ----

    def log_governance_event(self, decision_dict: Dict[str, Any]) -> Optional[AuditEntry]:
        """
        Log a governance decision to the audit chain.

        Args:
            decision_dict: GovernanceDecision.to_dict() output

        Returns:
            AuditEntry or None if logging failed
        """
        return self.log_event("governance_decision", {
            "action_id": decision_dict.get("action_id"),
            "tool_name": decision_dict.get("classification", {}).get("tool_name"),
            "risk_level": decision_dict.get("classification", {}).get("level"),
            "status": decision_dict.get("status"),
            "approved_by": decision_dict.get("approved_by"),
            "reasons": decision_dict.get("classification", {}).get("reasons", []),
        })

    def verify_digest(self) -> Dict[str, Any]:
        """
        Verify external digest against main audit log (v8.5).

        Compares each digest line against the corresponding audit entry.
        Designed to be called by external verification processes.

        Returns:
            Dict with valid, entries_checked, mismatches
        """
        digest_path = os.path.join(os.path.dirname(self.log_path), "audit_digest.sha256")
        result = {"valid": True, "entries_checked": 0, "mismatches": []}

        try:
            if not os.path.exists(digest_path):
                return {**result, "error": "digest_not_found"}

            # Load all audit entry hashes by sequence
            audit_hashes = {}
            if os.path.exists(self.log_path):
                with open(self.log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                data = json.loads(line)
                                audit_hashes[data.get("sequence", 0)] = data.get("entry_hash", "")
                            except json.JSONDecodeError:
                                continue

            # Compare digest entries
            with open(digest_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("|", 3)
                    if len(parts) < 4:
                        continue
                    try:
                        seq = int(parts[0])
                        digest_hash = parts[3]
                        result["entries_checked"] += 1

                        audit_hash = audit_hashes.get(seq)
                        if audit_hash is None:
                            result["valid"] = False
                            result["mismatches"].append({
                                "sequence": seq, "reason": "missing_in_audit"
                            })
                        elif audit_hash != digest_hash:
                            result["valid"] = False
                            result["mismatches"].append({
                                "sequence": seq, "reason": "hash_mismatch"
                            })
                    except (ValueError, IndexError):
                        continue

        except IOError as e:
            result["valid"] = False
            result["error"] = str(e)

        return result

    def get_digest_tail(self, n: int = 20) -> List[str]:
        """Get last N lines from the external digest file."""
        digest_path = os.path.join(os.path.dirname(self.log_path), "audit_digest.sha256")
        try:
            if not os.path.exists(digest_path):
                return []
            with open(digest_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                return [line.strip() for line in lines[-n:] if line.strip()]
        except IOError:
            return []

    def get_governance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Aggregate governance activity from audit log (v8.5).

        Args:
            hours: Look back period in hours

        Returns:
            Dict with counts by risk level, status, and autonomy ratio
        """
        cutoff = time.time() - (hours * 3600)
        summary = {
            "period_hours": hours,
            "total": 0,
            "by_risk_level": {},
            "by_status": {},
            "autonomy_ratio": 0.0,
        }

        try:
            entries = self.get_all()
            gov_entries = [
                e for e in entries
                if e.event_type == "governance_decision" and e.timestamp >= cutoff
            ]

            summary["total"] = len(gov_entries)
            auto_count = 0

            for entry in gov_entries:
                risk = entry.payload.get("risk_level", "unknown")
                status = entry.payload.get("status", "unknown")
                approved_by = entry.payload.get("approved_by", "")

                summary["by_risk_level"][risk] = summary["by_risk_level"].get(risk, 0) + 1
                summary["by_status"][status] = summary["by_status"].get(status, 0) + 1

                if approved_by == "auto":
                    auto_count += 1

            if summary["total"] > 0:
                summary["autonomy_ratio"] = round(auto_count / summary["total"], 2)

        except Exception:
            pass

        return summary
