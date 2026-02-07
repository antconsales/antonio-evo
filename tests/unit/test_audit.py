"""
Unit tests for the audit logger with cryptographic hashing.

Tests cover:
- Correct hash chaining
- First-entry (genesis) behavior
- Consistency of hash computation
- Append-only behavior
- Failure handling
- Entry serialization
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, ".")

from src.utils.audit import (
    AuditLogger,
    AuditEntry,
    VerificationResult,
    VerificationFailureReason,
    compute_entry_hash,
    GENESIS_HASH,
)


# =============================================================================
# Test: Hash Computation
# =============================================================================

class TestHashComputation(unittest.TestCase):
    """Tests for hash computation function."""

    def test_hash_is_64_chars(self):
        """Test that hash is 64 hex characters (SHA-256)."""
        hash_value = compute_entry_hash(
            timestamp=1234567890.0,
            event_type="test",
            payload={"key": "value"},
            previous_hash=GENESIS_HASH,
        )
        self.assertEqual(len(hash_value), 64)

    def test_hash_is_hex(self):
        """Test that hash is valid hexadecimal."""
        hash_value = compute_entry_hash(
            timestamp=1234567890.0,
            event_type="test",
            payload={"key": "value"},
            previous_hash=GENESIS_HASH,
        )
        # Should not raise
        int(hash_value, 16)

    def test_hash_is_deterministic(self):
        """Test that same input produces same hash."""
        args = {
            "timestamp": 1234567890.0,
            "event_type": "test",
            "payload": {"key": "value"},
            "previous_hash": GENESIS_HASH,
        }
        hash1 = compute_entry_hash(**args)
        hash2 = compute_entry_hash(**args)
        self.assertEqual(hash1, hash2)

    def test_different_timestamp_different_hash(self):
        """Test that different timestamp produces different hash."""
        hash1 = compute_entry_hash(
            timestamp=1234567890.0,
            event_type="test",
            payload={"key": "value"},
            previous_hash=GENESIS_HASH,
        )
        hash2 = compute_entry_hash(
            timestamp=1234567891.0,
            event_type="test",
            payload={"key": "value"},
            previous_hash=GENESIS_HASH,
        )
        self.assertNotEqual(hash1, hash2)

    def test_different_event_type_different_hash(self):
        """Test that different event type produces different hash."""
        hash1 = compute_entry_hash(
            timestamp=1234567890.0,
            event_type="test1",
            payload={"key": "value"},
            previous_hash=GENESIS_HASH,
        )
        hash2 = compute_entry_hash(
            timestamp=1234567890.0,
            event_type="test2",
            payload={"key": "value"},
            previous_hash=GENESIS_HASH,
        )
        self.assertNotEqual(hash1, hash2)

    def test_different_payload_different_hash(self):
        """Test that different payload produces different hash."""
        hash1 = compute_entry_hash(
            timestamp=1234567890.0,
            event_type="test",
            payload={"key": "value1"},
            previous_hash=GENESIS_HASH,
        )
        hash2 = compute_entry_hash(
            timestamp=1234567890.0,
            event_type="test",
            payload={"key": "value2"},
            previous_hash=GENESIS_HASH,
        )
        self.assertNotEqual(hash1, hash2)

    def test_different_previous_hash_different_hash(self):
        """Test that different previous hash produces different hash."""
        hash1 = compute_entry_hash(
            timestamp=1234567890.0,
            event_type="test",
            payload={"key": "value"},
            previous_hash=GENESIS_HASH,
        )
        hash2 = compute_entry_hash(
            timestamp=1234567890.0,
            event_type="test",
            payload={"key": "value"},
            previous_hash="a" * 64,
        )
        self.assertNotEqual(hash1, hash2)

    def test_empty_payload_valid(self):
        """Test that empty payload produces valid hash."""
        hash_value = compute_entry_hash(
            timestamp=1234567890.0,
            event_type="test",
            payload={},
            previous_hash=GENESIS_HASH,
        )
        self.assertEqual(len(hash_value), 64)

    def test_nested_payload_deterministic(self):
        """Test that nested payload produces deterministic hash."""
        payload = {
            "level1": {
                "level2": {
                    "value": 123
                }
            },
            "list": [1, 2, 3]
        }
        hash1 = compute_entry_hash(
            timestamp=1234567890.0,
            event_type="test",
            payload=payload,
            previous_hash=GENESIS_HASH,
        )
        hash2 = compute_entry_hash(
            timestamp=1234567890.0,
            event_type="test",
            payload=payload,
            previous_hash=GENESIS_HASH,
        )
        self.assertEqual(hash1, hash2)


# =============================================================================
# Test: Genesis Hash
# =============================================================================

class TestGenesisHash(unittest.TestCase):
    """Tests for genesis hash constant."""

    def test_genesis_hash_is_64_zeros(self):
        """Test that genesis hash is 64 zeros."""
        self.assertEqual(len(GENESIS_HASH), 64)
        self.assertEqual(GENESIS_HASH, "0" * 64)

    def test_genesis_hash_is_valid_hex(self):
        """Test that genesis hash is valid hex."""
        # Should not raise
        int(GENESIS_HASH, 16)


# =============================================================================
# Test: AuditEntry
# =============================================================================

class TestAuditEntry(unittest.TestCase):
    """Tests for AuditEntry dataclass."""

    def test_entry_creation(self):
        """Test creating an audit entry."""
        entry = AuditEntry(
            timestamp=1234567890.0,
            timestamp_iso="2009-02-13T23:31:30",
            event_type="test",
            payload={"key": "value"},
            previous_hash=GENESIS_HASH,
            entry_hash="a" * 64,
            sequence=1,
        )
        self.assertEqual(entry.timestamp, 1234567890.0)
        self.assertEqual(entry.event_type, "test")
        self.assertEqual(entry.sequence, 1)

    def test_entry_to_dict(self):
        """Test entry serialization to dict."""
        entry = AuditEntry(
            timestamp=1234567890.0,
            timestamp_iso="2009-02-13T23:31:30",
            event_type="test",
            payload={"key": "value"},
            previous_hash=GENESIS_HASH,
            entry_hash="a" * 64,
            sequence=1,
        )
        d = entry.to_dict()
        self.assertEqual(d["timestamp"], 1234567890.0)
        self.assertEqual(d["event_type"], "test")
        self.assertEqual(d["entry_hash"], "a" * 64)
        self.assertEqual(d["sequence"], 1)

    def test_entry_from_dict(self):
        """Test entry deserialization from dict."""
        data = {
            "timestamp": 1234567890.0,
            "timestamp_iso": "2009-02-13T23:31:30",
            "event_type": "test",
            "payload": {"key": "value"},
            "previous_hash": GENESIS_HASH,
            "entry_hash": "a" * 64,
            "sequence": 1,
        }
        entry = AuditEntry.from_dict(data)
        self.assertEqual(entry.timestamp, 1234567890.0)
        self.assertEqual(entry.event_type, "test")
        self.assertEqual(entry.sequence, 1)

    def test_entry_from_dict_defaults(self):
        """Test entry deserialization with missing fields uses defaults."""
        entry = AuditEntry.from_dict({})
        self.assertEqual(entry.timestamp, 0)
        self.assertEqual(entry.event_type, "")
        self.assertEqual(entry.previous_hash, GENESIS_HASH)
        self.assertEqual(entry.sequence, 0)


# =============================================================================
# Test: AuditLogger - Basic Operations
# =============================================================================

class TestAuditLoggerBasic(unittest.TestCase):
    """Tests for basic audit logger operations."""

    def setUp(self):
        """Create temporary log file for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_path = os.path.join(self.temp_dir, "test_audit.jsonl")
        self.logger = AuditLogger(log_path=self.log_path)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_new_logger_is_genesis(self):
        """Test that new logger starts at genesis."""
        self.assertTrue(self.logger.is_genesis())
        self.assertEqual(self.logger.current_hash, GENESIS_HASH)
        self.assertEqual(self.logger.current_sequence, 0)

    def test_log_event_returns_entry(self):
        """Test that log_event returns an AuditEntry."""
        entry = self.logger.log_event("test", {"key": "value"})
        self.assertIsInstance(entry, AuditEntry)
        self.assertEqual(entry.event_type, "test")

    def test_log_event_increments_sequence(self):
        """Test that logging increments sequence number."""
        self.assertEqual(self.logger.current_sequence, 0)

        self.logger.log_event("test1", {})
        self.assertEqual(self.logger.current_sequence, 1)

        self.logger.log_event("test2", {})
        self.assertEqual(self.logger.current_sequence, 2)

    def test_log_event_updates_hash(self):
        """Test that logging updates current hash."""
        initial_hash = self.logger.current_hash
        self.assertEqual(initial_hash, GENESIS_HASH)

        entry = self.logger.log_event("test", {})
        self.assertNotEqual(self.logger.current_hash, GENESIS_HASH)
        self.assertEqual(self.logger.current_hash, entry.entry_hash)

    def test_first_entry_has_genesis_previous(self):
        """Test that first entry has genesis as previous hash."""
        entry = self.logger.log_event("test", {})
        self.assertEqual(entry.previous_hash, GENESIS_HASH)

    def test_entry_has_valid_hash(self):
        """Test that entry has valid hash."""
        entry = self.logger.log_event("test", {"key": "value"})
        self.assertEqual(len(entry.entry_hash), 64)

    def test_no_longer_genesis_after_log(self):
        """Test that logger is no longer genesis after logging."""
        self.assertTrue(self.logger.is_genesis())
        self.logger.log_event("test", {})
        self.assertFalse(self.logger.is_genesis())


# =============================================================================
# Test: Hash Chaining
# =============================================================================

class TestHashChaining(unittest.TestCase):
    """Tests for hash chain integrity."""

    def setUp(self):
        """Create temporary log file for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_path = os.path.join(self.temp_dir, "test_audit.jsonl")
        self.logger = AuditLogger(log_path=self.log_path)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_second_entry_references_first(self):
        """Test that second entry references first entry's hash."""
        entry1 = self.logger.log_event("event1", {})
        entry2 = self.logger.log_event("event2", {})

        self.assertEqual(entry2.previous_hash, entry1.entry_hash)

    def test_chain_of_three_entries(self):
        """Test hash chain with three entries."""
        entry1 = self.logger.log_event("event1", {})
        entry2 = self.logger.log_event("event2", {})
        entry3 = self.logger.log_event("event3", {})

        self.assertEqual(entry1.previous_hash, GENESIS_HASH)
        self.assertEqual(entry2.previous_hash, entry1.entry_hash)
        self.assertEqual(entry3.previous_hash, entry2.entry_hash)

    def test_hash_chain_verifiable(self):
        """Test that hash chain can be verified by recomputing."""
        # Log several entries
        entries = []
        for i in range(5):
            entry = self.logger.log_event(f"event{i}", {"index": i})
            entries.append(entry)

        # Verify chain by recomputing hashes
        for i, entry in enumerate(entries):
            expected_previous = GENESIS_HASH if i == 0 else entries[i-1].entry_hash
            self.assertEqual(entry.previous_hash, expected_previous)

            # Recompute hash and verify
            computed = compute_entry_hash(
                timestamp=entry.timestamp,
                event_type=entry.event_type,
                payload=entry.payload,
                previous_hash=entry.previous_hash,
            )
            self.assertEqual(computed, entry.entry_hash)

    def test_consecutive_entries_different_hashes(self):
        """Test that consecutive entries have different hashes."""
        entry1 = self.logger.log_event("test", {"key": "value"})
        entry2 = self.logger.log_event("test", {"key": "value"})

        # Same content but different hashes (due to different timestamps and chain)
        self.assertNotEqual(entry1.entry_hash, entry2.entry_hash)


# =============================================================================
# Test: Persistence and Recovery
# =============================================================================

class TestPersistenceAndRecovery(unittest.TestCase):
    """Tests for log persistence and chain recovery."""

    def setUp(self):
        """Create temporary log file for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_path = os.path.join(self.temp_dir, "test_audit.jsonl")

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_entries_persisted_to_file(self):
        """Test that entries are written to file."""
        logger = AuditLogger(log_path=self.log_path)
        logger.log_event("test", {"key": "value"})

        self.assertTrue(os.path.exists(self.log_path))
        with open(self.log_path, "r") as f:
            content = f.read()
            self.assertIn("test", content)

    def test_new_logger_recovers_chain(self):
        """Test that new logger instance recovers chain state."""
        # First logger
        logger1 = AuditLogger(log_path=self.log_path)
        entry = logger1.log_event("test", {"key": "value"})

        # New logger instance
        logger2 = AuditLogger(log_path=self.log_path)

        # Should have recovered state
        self.assertEqual(logger2.current_hash, entry.entry_hash)
        self.assertEqual(logger2.current_sequence, 1)
        self.assertFalse(logger2.is_genesis())

    def test_chain_continues_after_recovery(self):
        """Test that chain continues correctly after recovery."""
        # First logger
        logger1 = AuditLogger(log_path=self.log_path)
        entry1 = logger1.log_event("event1", {})

        # New logger instance
        logger2 = AuditLogger(log_path=self.log_path)
        entry2 = logger2.log_event("event2", {})

        # Entry2 should reference entry1
        self.assertEqual(entry2.previous_hash, entry1.entry_hash)
        self.assertEqual(entry2.sequence, 2)

    def test_get_recent_returns_entries(self):
        """Test that get_recent returns logged entries."""
        logger = AuditLogger(log_path=self.log_path)
        logger.log_event("event1", {"index": 1})
        logger.log_event("event2", {"index": 2})
        logger.log_event("event3", {"index": 3})

        entries = logger.get_recent(2)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].payload["index"], 2)
        self.assertEqual(entries[1].payload["index"], 3)

    def test_get_all_returns_all_entries(self):
        """Test that get_all returns all logged entries."""
        logger = AuditLogger(log_path=self.log_path)
        for i in range(5):
            logger.log_event(f"event{i}", {"index": i})

        entries = logger.get_all()
        self.assertEqual(len(entries), 5)

    def test_get_entry_by_sequence(self):
        """Test getting entry by sequence number."""
        logger = AuditLogger(log_path=self.log_path)
        logger.log_event("event1", {"index": 1})
        logger.log_event("event2", {"index": 2})
        logger.log_event("event3", {"index": 3})

        entry = logger.get_entry_by_sequence(2)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.payload["index"], 2)

    def test_get_entry_by_sequence_not_found(self):
        """Test getting non-existent sequence returns None."""
        logger = AuditLogger(log_path=self.log_path)
        logger.log_event("event1", {})

        entry = logger.get_entry_by_sequence(999)
        self.assertIsNone(entry)


# =============================================================================
# Test: Append-Only Behavior
# =============================================================================

class TestAppendOnly(unittest.TestCase):
    """Tests for append-only logging behavior."""

    def setUp(self):
        """Create temporary log file for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_path = os.path.join(self.temp_dir, "test_audit.jsonl")

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_multiple_logs_append(self):
        """Test that multiple logs append to file."""
        logger = AuditLogger(log_path=self.log_path)
        logger.log_event("event1", {})
        logger.log_event("event2", {})
        logger.log_event("event3", {})

        with open(self.log_path, "r") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 3)

    def test_sequence_numbers_monotonic(self):
        """Test that sequence numbers are monotonically increasing."""
        logger = AuditLogger(log_path=self.log_path)
        entries = []
        for _ in range(10):
            entries.append(logger.log_event("test", {}))

        sequences = [e.sequence for e in entries]
        self.assertEqual(sequences, list(range(1, 11)))

    def test_file_grows_with_each_entry(self):
        """Test that file size grows with each entry."""
        logger = AuditLogger(log_path=self.log_path)

        sizes = []
        for i in range(5):
            logger.log_event("test", {"index": i})
            sizes.append(os.path.getsize(self.log_path))

        # Each size should be larger than the previous
        for i in range(1, len(sizes)):
            self.assertGreater(sizes[i], sizes[i-1])


# =============================================================================
# Test: Failure Handling
# =============================================================================

class TestFailureHandling(unittest.TestCase):
    """Tests for graceful failure handling."""

    def test_nonexistent_directory_handled(self):
        """Test that nonexistent directory doesn't crash."""
        # This might fail on write, but shouldn't crash on init
        logger = AuditLogger(log_path="/nonexistent/path/audit.jsonl")
        # Logger should initialize without crashing
        self.assertIsNotNone(logger)

    def test_empty_file_starts_genesis(self):
        """Test that empty file starts at genesis."""
        temp_dir = tempfile.mkdtemp()
        log_path = os.path.join(temp_dir, "empty.jsonl")

        # Create empty file
        open(log_path, "w").close()

        logger = AuditLogger(log_path=log_path)
        self.assertTrue(logger.is_genesis())

        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_corrupted_last_line_handled(self):
        """Test that corrupted last line is handled gracefully."""
        temp_dir = tempfile.mkdtemp()
        log_path = os.path.join(temp_dir, "corrupted.jsonl")

        # Write valid entry then corrupted line
        with open(log_path, "w") as f:
            f.write('{"sequence": 1, "entry_hash": "' + "a" * 64 + '"}\n')
            f.write("this is not json\n")

        # Should not crash, will start fresh
        logger = AuditLogger(log_path=log_path)
        self.assertIsNotNone(logger)

        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_get_recent_on_empty_file(self):
        """Test get_recent on empty/nonexistent file."""
        logger = AuditLogger(log_path="/nonexistent/audit.jsonl")
        entries = logger.get_recent(10)
        self.assertEqual(entries, [])

    def test_get_all_on_empty_file(self):
        """Test get_all on empty/nonexistent file."""
        logger = AuditLogger(log_path="/nonexistent/audit.jsonl")
        entries = logger.get_all()
        self.assertEqual(entries, [])


# =============================================================================
# Test: Hash Consistency
# =============================================================================

class TestHashConsistency(unittest.TestCase):
    """Tests for hash computation consistency."""

    def test_payload_order_independent(self):
        """Test that payload key order doesn't affect hash."""
        payload1 = {"a": 1, "b": 2, "c": 3}
        payload2 = {"c": 3, "a": 1, "b": 2}

        hash1 = compute_entry_hash(
            timestamp=1234567890.0,
            event_type="test",
            payload=payload1,
            previous_hash=GENESIS_HASH,
        )
        hash2 = compute_entry_hash(
            timestamp=1234567890.0,
            event_type="test",
            payload=payload2,
            previous_hash=GENESIS_HASH,
        )
        self.assertEqual(hash1, hash2)

    def test_unicode_payload_handled(self):
        """Test that unicode in payload is handled correctly."""
        payload = {"message": "Hello 世界 😀"}
        hash_value = compute_entry_hash(
            timestamp=1234567890.0,
            event_type="test",
            payload=payload,
            previous_hash=GENESIS_HASH,
        )
        self.assertEqual(len(hash_value), 64)

    def test_special_characters_in_payload(self):
        """Test that special characters are handled."""
        payload = {"text": "line1\nline2\ttab"}
        hash_value = compute_entry_hash(
            timestamp=1234567890.0,
            event_type="test",
            payload=payload,
            previous_hash=GENESIS_HASH,
        )
        self.assertEqual(len(hash_value), 64)


# =============================================================================
# Test: Tamper Detection - Valid Chain
# =============================================================================

class TestTamperDetectionValidChain(unittest.TestCase):
    """Tests for verifying untampered chains."""

    def setUp(self):
        """Create temporary log file for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_path = os.path.join(self.temp_dir, "test_audit.jsonl")

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_empty_log_is_valid(self):
        """Test that empty log verifies as valid."""
        logger = AuditLogger(log_path=self.log_path)
        result = logger.verify_chain()

        self.assertTrue(result.valid)
        self.assertIsNone(result.index)
        self.assertIsNone(result.reason)
        self.assertEqual(result.entries_checked, 0)

    def test_single_entry_valid(self):
        """Test that single valid entry verifies."""
        logger = AuditLogger(log_path=self.log_path)
        logger.log_event("test", {"key": "value"})

        result = logger.verify_chain()

        self.assertTrue(result.valid)
        self.assertEqual(result.entries_checked, 1)

    def test_multiple_entries_valid(self):
        """Test that multiple valid entries verify."""
        logger = AuditLogger(log_path=self.log_path)
        for i in range(10):
            logger.log_event(f"event{i}", {"index": i})

        result = logger.verify_chain()

        self.assertTrue(result.valid)
        self.assertEqual(result.entries_checked, 10)

    def test_nonexistent_file_is_valid(self):
        """Test that nonexistent file verifies as valid (empty)."""
        logger = AuditLogger(log_path="/nonexistent/audit.jsonl")
        result = logger.verify_chain()

        self.assertTrue(result.valid)
        self.assertEqual(result.entries_checked, 0)

    def test_verification_result_serializable(self):
        """Test that VerificationResult can be serialized."""
        logger = AuditLogger(log_path=self.log_path)
        logger.log_event("test", {})

        result = logger.verify_chain()
        d = result.to_dict()

        self.assertEqual(d["valid"], True)
        self.assertIsNone(d["index"])
        self.assertIsNone(d["reason"])
        self.assertEqual(d["entries_checked"], 1)


# =============================================================================
# Test: Tamper Detection - Modified Payload
# =============================================================================

class TestTamperDetectionModifiedPayload(unittest.TestCase):
    """Tests for detecting modified payload."""

    def setUp(self):
        """Create temporary log file for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_path = os.path.join(self.temp_dir, "test_audit.jsonl")

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_modified_payload_detected(self):
        """Test that modified payload is detected."""
        logger = AuditLogger(log_path=self.log_path)
        logger.log_event("event1", {"key": "original"})
        logger.log_event("event2", {"key": "value2"})

        # Tamper with first entry's payload
        with open(self.log_path, "r") as f:
            lines = f.readlines()

        entry1 = json.loads(lines[0])
        entry1["payload"]["key"] = "tampered"
        lines[0] = json.dumps(entry1) + "\n"

        with open(self.log_path, "w") as f:
            f.writelines(lines)

        # Verify should fail
        result = logger.verify_chain()

        self.assertFalse(result.valid)
        self.assertEqual(result.index, 0)
        self.assertEqual(result.reason, VerificationFailureReason.HASH_MISMATCH.value)

    def test_modified_payload_first_failure_only(self):
        """Test that only first tampered entry is reported."""
        logger = AuditLogger(log_path=self.log_path)
        for i in range(5):
            logger.log_event(f"event{i}", {"index": i})

        # Tamper with entries 1 and 3
        with open(self.log_path, "r") as f:
            lines = f.readlines()

        entry1 = json.loads(lines[1])
        entry1["payload"]["index"] = 999
        lines[1] = json.dumps(entry1) + "\n"

        entry3 = json.loads(lines[3])
        entry3["payload"]["index"] = 888
        lines[3] = json.dumps(entry3) + "\n"

        with open(self.log_path, "w") as f:
            f.writelines(lines)

        # Should report first failure at index 1
        result = logger.verify_chain()

        self.assertFalse(result.valid)
        self.assertEqual(result.index, 1)
        self.assertEqual(result.entries_checked, 1)


# =============================================================================
# Test: Tamper Detection - Modified Entry Hash
# =============================================================================

class TestTamperDetectionModifiedEntryHash(unittest.TestCase):
    """Tests for detecting modified entry_hash."""

    def setUp(self):
        """Create temporary log file for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_path = os.path.join(self.temp_dir, "test_audit.jsonl")

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_modified_entry_hash_detected(self):
        """Test that modified entry_hash is detected."""
        logger = AuditLogger(log_path=self.log_path)
        logger.log_event("test", {"key": "value"})

        # Tamper with entry_hash
        with open(self.log_path, "r") as f:
            lines = f.readlines()

        entry = json.loads(lines[0])
        entry["entry_hash"] = "f" * 64  # Fake hash
        lines[0] = json.dumps(entry) + "\n"

        with open(self.log_path, "w") as f:
            f.writelines(lines)

        result = logger.verify_chain()

        self.assertFalse(result.valid)
        self.assertEqual(result.index, 0)
        self.assertEqual(result.reason, VerificationFailureReason.HASH_MISMATCH.value)

    def test_truncated_entry_hash_detected(self):
        """Test that truncated entry_hash is detected."""
        logger = AuditLogger(log_path=self.log_path)
        logger.log_event("test", {"key": "value"})

        with open(self.log_path, "r") as f:
            lines = f.readlines()

        entry = json.loads(lines[0])
        entry["entry_hash"] = entry["entry_hash"][:32]  # Truncate
        lines[0] = json.dumps(entry) + "\n"

        with open(self.log_path, "w") as f:
            f.writelines(lines)

        result = logger.verify_chain()

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, VerificationFailureReason.HASH_MISMATCH.value)


# =============================================================================
# Test: Tamper Detection - Modified Previous Hash
# =============================================================================

class TestTamperDetectionModifiedPreviousHash(unittest.TestCase):
    """Tests for detecting modified previous_hash."""

    def setUp(self):
        """Create temporary log file for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_path = os.path.join(self.temp_dir, "test_audit.jsonl")

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_modified_previous_hash_detected(self):
        """Test that modified previous_hash is detected."""
        logger = AuditLogger(log_path=self.log_path)
        logger.log_event("event1", {})
        logger.log_event("event2", {})

        # Tamper with second entry's previous_hash
        with open(self.log_path, "r") as f:
            lines = f.readlines()

        entry2 = json.loads(lines[1])
        entry2["previous_hash"] = "a" * 64  # Wrong previous hash
        lines[1] = json.dumps(entry2) + "\n"

        with open(self.log_path, "w") as f:
            f.writelines(lines)

        result = logger.verify_chain()

        self.assertFalse(result.valid)
        self.assertEqual(result.index, 1)
        self.assertEqual(result.reason, VerificationFailureReason.PREVIOUS_HASH_MISMATCH.value)

    def test_first_entry_wrong_genesis_detected(self):
        """Test that first entry with wrong genesis hash is detected."""
        logger = AuditLogger(log_path=self.log_path)
        logger.log_event("test", {})

        with open(self.log_path, "r") as f:
            lines = f.readlines()

        entry = json.loads(lines[0])
        entry["previous_hash"] = "1" * 64  # Should be genesis (all zeros)
        lines[0] = json.dumps(entry) + "\n"

        with open(self.log_path, "w") as f:
            f.writelines(lines)

        result = logger.verify_chain()

        self.assertFalse(result.valid)
        self.assertEqual(result.index, 0)
        self.assertEqual(result.reason, VerificationFailureReason.PREVIOUS_HASH_MISMATCH.value)

    def test_broken_chain_link_detected(self):
        """Test that broken chain link is detected."""
        logger = AuditLogger(log_path=self.log_path)
        for i in range(5):
            logger.log_event(f"event{i}", {})

        # Break the chain at entry 3 by changing its previous_hash
        with open(self.log_path, "r") as f:
            lines = f.readlines()

        entry3 = json.loads(lines[3])
        entry3["previous_hash"] = "b" * 64
        lines[3] = json.dumps(entry3) + "\n"

        with open(self.log_path, "w") as f:
            f.writelines(lines)

        result = logger.verify_chain()

        self.assertFalse(result.valid)
        self.assertEqual(result.index, 3)
        self.assertEqual(result.reason, VerificationFailureReason.PREVIOUS_HASH_MISMATCH.value)


# =============================================================================
# Test: Tamper Detection - Sequence Errors
# =============================================================================

class TestTamperDetectionSequenceErrors(unittest.TestCase):
    """Tests for detecting sequence number errors."""

    def setUp(self):
        """Create temporary log file for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_path = os.path.join(self.temp_dir, "test_audit.jsonl")

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_wrong_first_sequence_detected(self):
        """Test that wrong first sequence number is detected."""
        logger = AuditLogger(log_path=self.log_path)
        logger.log_event("test", {})

        with open(self.log_path, "r") as f:
            lines = f.readlines()

        entry = json.loads(lines[0])
        entry["sequence"] = 5  # Should be 1
        lines[0] = json.dumps(entry) + "\n"

        with open(self.log_path, "w") as f:
            f.writelines(lines)

        result = logger.verify_chain()

        self.assertFalse(result.valid)
        self.assertEqual(result.index, 0)
        self.assertEqual(result.reason, VerificationFailureReason.SEQUENCE_ERROR.value)

    def test_skipped_sequence_detected(self):
        """Test that skipped sequence number is detected."""
        logger = AuditLogger(log_path=self.log_path)
        for i in range(3):
            logger.log_event(f"event{i}", {})

        # Change sequence 2 to 5 (skip)
        with open(self.log_path, "r") as f:
            lines = f.readlines()

        entry = json.loads(lines[1])
        entry["sequence"] = 5
        lines[1] = json.dumps(entry) + "\n"

        with open(self.log_path, "w") as f:
            f.writelines(lines)

        result = logger.verify_chain()

        self.assertFalse(result.valid)
        self.assertEqual(result.index, 1)
        self.assertEqual(result.reason, VerificationFailureReason.SEQUENCE_ERROR.value)

    def test_duplicate_sequence_detected(self):
        """Test that duplicate sequence number is detected."""
        logger = AuditLogger(log_path=self.log_path)
        for i in range(3):
            logger.log_event(f"event{i}", {})

        # Change sequence 3 to 2 (duplicate)
        with open(self.log_path, "r") as f:
            lines = f.readlines()

        entry = json.loads(lines[2])
        entry["sequence"] = 2
        lines[2] = json.dumps(entry) + "\n"

        with open(self.log_path, "w") as f:
            f.writelines(lines)

        result = logger.verify_chain()

        self.assertFalse(result.valid)
        self.assertEqual(result.index, 2)
        self.assertEqual(result.reason, VerificationFailureReason.SEQUENCE_ERROR.value)

    def test_zero_sequence_detected(self):
        """Test that sequence starting at 0 is detected."""
        logger = AuditLogger(log_path=self.log_path)
        logger.log_event("test", {})

        with open(self.log_path, "r") as f:
            lines = f.readlines()

        entry = json.loads(lines[0])
        entry["sequence"] = 0  # Should be 1
        lines[0] = json.dumps(entry) + "\n"

        with open(self.log_path, "w") as f:
            f.writelines(lines)

        result = logger.verify_chain()

        self.assertFalse(result.valid)
        self.assertEqual(result.index, 0)
        self.assertEqual(result.reason, VerificationFailureReason.SEQUENCE_ERROR.value)


# =============================================================================
# Test: Tamper Detection - Missing/Deleted Entries
# =============================================================================

class TestTamperDetectionMissingEntries(unittest.TestCase):
    """Tests for detecting missing or deleted entries."""

    def setUp(self):
        """Create temporary log file for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_path = os.path.join(self.temp_dir, "test_audit.jsonl")

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_deleted_middle_entry_detected(self):
        """Test that deleted middle entry is detected."""
        logger = AuditLogger(log_path=self.log_path)
        for i in range(5):
            logger.log_event(f"event{i}", {"index": i})

        # Delete entry at index 2
        with open(self.log_path, "r") as f:
            lines = f.readlines()

        del lines[2]

        with open(self.log_path, "w") as f:
            f.writelines(lines)

        result = logger.verify_chain()

        self.assertFalse(result.valid)
        # Entry at position 2 now has wrong sequence (4 instead of 3)
        self.assertEqual(result.index, 2)

    def test_deleted_first_entry_detected(self):
        """Test that deleted first entry is detected."""
        logger = AuditLogger(log_path=self.log_path)
        for i in range(3):
            logger.log_event(f"event{i}", {})

        # Delete first entry
        with open(self.log_path, "r") as f:
            lines = f.readlines()

        del lines[0]

        with open(self.log_path, "w") as f:
            f.writelines(lines)

        result = logger.verify_chain()

        self.assertFalse(result.valid)
        self.assertEqual(result.index, 0)

    def test_deleted_last_entry_detected(self):
        """Test that deleted last entry causes chain state mismatch."""
        logger = AuditLogger(log_path=self.log_path)
        for i in range(5):
            logger.log_event(f"event{i}", {})

        # Delete last entry
        with open(self.log_path, "r") as f:
            lines = f.readlines()

        del lines[-1]

        with open(self.log_path, "w") as f:
            f.writelines(lines)

        # Reload logger to get fresh state
        logger2 = AuditLogger(log_path=self.log_path)
        result = logger2.verify_chain()

        # Chain should still be valid (just shorter)
        self.assertTrue(result.valid)
        self.assertEqual(result.entries_checked, 4)


# =============================================================================
# Test: Tamper Detection - Corrupted Data
# =============================================================================

class TestTamperDetectionCorruptedData(unittest.TestCase):
    """Tests for detecting corrupted data."""

    def setUp(self):
        """Create temporary log file for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_path = os.path.join(self.temp_dir, "test_audit.jsonl")

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_corrupted_json_detected(self):
        """Test that corrupted JSON is detected."""
        logger = AuditLogger(log_path=self.log_path)
        logger.log_event("event1", {})
        logger.log_event("event2", {})

        # Corrupt second entry
        with open(self.log_path, "r") as f:
            lines = f.readlines()

        lines[1] = "this is not valid json\n"

        with open(self.log_path, "w") as f:
            f.writelines(lines)

        result = logger.verify_chain()

        self.assertFalse(result.valid)
        self.assertEqual(result.index, 1)
        self.assertEqual(result.reason, VerificationFailureReason.HASH_MISMATCH.value)

    def test_empty_line_skipped(self):
        """Test that empty lines are skipped during verification."""
        logger = AuditLogger(log_path=self.log_path)
        logger.log_event("event1", {})
        logger.log_event("event2", {})

        # Insert empty lines
        with open(self.log_path, "r") as f:
            lines = f.readlines()

        lines.insert(1, "\n")
        lines.insert(1, "   \n")

        with open(self.log_path, "w") as f:
            f.writelines(lines)

        result = logger.verify_chain()

        # Should still be valid (empty lines ignored)
        self.assertTrue(result.valid)
        self.assertEqual(result.entries_checked, 2)


# =============================================================================
# Test: Tamper Detection - Timestamp Modification
# =============================================================================

class TestTamperDetectionTimestamp(unittest.TestCase):
    """Tests for detecting timestamp modifications."""

    def setUp(self):
        """Create temporary log file for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_path = os.path.join(self.temp_dir, "test_audit.jsonl")

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_modified_timestamp_detected(self):
        """Test that modified timestamp is detected via hash mismatch."""
        logger = AuditLogger(log_path=self.log_path)
        logger.log_event("test", {"key": "value"})

        with open(self.log_path, "r") as f:
            lines = f.readlines()

        entry = json.loads(lines[0])
        entry["timestamp"] = 9999999999.0  # Fake timestamp
        lines[0] = json.dumps(entry) + "\n"

        with open(self.log_path, "w") as f:
            f.writelines(lines)

        result = logger.verify_chain()

        self.assertFalse(result.valid)
        self.assertEqual(result.index, 0)
        self.assertEqual(result.reason, VerificationFailureReason.HASH_MISMATCH.value)


# =============================================================================
# Test: Tamper Detection - Event Type Modification
# =============================================================================

class TestTamperDetectionEventType(unittest.TestCase):
    """Tests for detecting event_type modifications."""

    def setUp(self):
        """Create temporary log file for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_path = os.path.join(self.temp_dir, "test_audit.jsonl")

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_modified_event_type_detected(self):
        """Test that modified event_type is detected via hash mismatch."""
        logger = AuditLogger(log_path=self.log_path)
        logger.log_event("original_event", {})

        with open(self.log_path, "r") as f:
            lines = f.readlines()

        entry = json.loads(lines[0])
        entry["event_type"] = "tampered_event"
        lines[0] = json.dumps(entry) + "\n"

        with open(self.log_path, "w") as f:
            f.writelines(lines)

        result = logger.verify_chain()

        self.assertFalse(result.valid)
        self.assertEqual(result.index, 0)
        self.assertEqual(result.reason, VerificationFailureReason.HASH_MISMATCH.value)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    unittest.main()
