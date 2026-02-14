"""
Governance Engine (v8.5) — Deterministic risk classification and approval gating.

Every tool call passes through this engine before execution.
Risk rules are in config/governance.json, NOT in the LLM.

Principle: Code classifies, code gates, code logs.
The LLM never decides its own risk level.

Risk Levels:
- LOW: Auto-execute immediately (70% of actions)
- MEDIUM: Auto-execute with logging (read-modify operations)
- HIGH: Require explicit human approval (code execution, installs)
- CRITICAL: Require approval + confirmation (destructive, credential access)
"""

import json
import logging
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk classification levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


RISK_ORDER = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2, RiskLevel.CRITICAL: 3}


@dataclass
class RiskClassification:
    """Result of risk analysis for a tool call."""
    level: RiskLevel
    tool_name: str
    reasons: List[str]
    constraints: Dict[str, Any] = field(default_factory=dict)
    requires_approval: bool = False
    auto_approve_after_secs: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "tool_name": self.tool_name,
            "reasons": self.reasons,
            "constraints": self.constraints,
            "requires_approval": self.requires_approval,
            "auto_approve_after_secs": self.auto_approve_after_secs,
        }


@dataclass
class GovernanceDecision:
    """Decision on whether a tool call may proceed."""
    action_id: str
    classification: RiskClassification
    status: str = "pending"  # pending, approved, denied, expired, executed, failed
    approved_by: Optional[str] = None  # auto, user, timeout
    approved_at: Optional[float] = None
    constraints_applied: Dict[str, Any] = field(default_factory=dict)
    execution_result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "classification": self.classification.to_dict(),
            "status": self.status,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "constraints_applied": self.constraints_applied,
        }


class GovernanceEngine:
    """
    Deterministic risk classification and approval gating.

    All tool calls are classified by risk level based on:
    1. Base tool risk (from config)
    2. Argument pattern escalation (regex on arguments)
    3. Rate-based escalation (too many calls/minute)

    LOW/MEDIUM → auto-approved. HIGH/CRITICAL → require human approval.
    """

    def __init__(
        self,
        db_path: str = "data/evomemory.db",
        config_path: str = "config/governance.json",
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()

        # Load configuration
        self._config = self._load_config(config_path)
        self._tool_risk = {}
        self._escalation_rules = []
        self._auto_approve = {}
        self._rate_escalation = {}
        self._parse_config()

        # Compile escalation regexes once
        self._compiled_rules = []
        for rule in self._escalation_rules:
            try:
                self._compiled_rules.append({
                    "regex": re.compile(rule["pattern"], re.IGNORECASE),
                    "field": rule.get("field", "*"),
                    "escalate_to": RiskLevel(rule["escalate_to"]),
                })
            except (re.error, ValueError):
                logger.warning(f"Invalid escalation rule: {rule}")

        # Rate tracking
        self._rate_lock = threading.Lock()
        self._action_timestamps: List[float] = []

        # Safe Degradation Mode (v8.5)
        self._dashboard = None
        self._audit = None
        self._system_mode_cache: Optional[Dict[str, Any]] = None
        self._system_mode_cache_ts: float = 0.0
        self._system_mode_cache_ttl: float = 60.0  # Re-check every 60s

        self._init_schema()
        logger.info(f"Governance Engine initialized ({len(self._tool_risk)} tool rules, "
                    f"{len(self._compiled_rules)} escalation rules)")

    def _load_config(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Governance config not found ({path}), using defaults: {e}")
            return {}

    def _parse_config(self):
        self._tool_risk = {
            name: RiskLevel(level)
            for name, level in self._config.get("tool_risk_levels", {}).items()
        }
        self._escalation_rules = self._config.get("argument_escalation_rules", [])
        self._auto_approve = self._config.get("auto_approve", {
            "low": True, "medium_delay_secs": 0, "high": False, "critical": False,
        })
        self._rate_escalation = self._config.get("rate_escalation", {
            "max_actions_per_minute": 15, "escalate_to": "medium",
        })

    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path), check_same_thread=False, timeout=30.0
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_schema(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS governance_actions (
                    action_id TEXT PRIMARY KEY,
                    plan_id TEXT,
                    tool_name TEXT NOT NULL,
                    arguments_json TEXT,
                    risk_level TEXT NOT NULL,
                    risk_reasons TEXT,
                    constraints_json TEXT,
                    status TEXT DEFAULT 'pending',
                    approved_by TEXT,
                    denied_reason TEXT,
                    created_at REAL NOT NULL,
                    approved_at REAL,
                    executed_at REAL,
                    result_json TEXT,
                    session_id TEXT
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_gov_status
                ON governance_actions(status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_gov_risk
                ON governance_actions(risk_level)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_gov_created
                ON governance_actions(created_at DESC)
            """)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    # ---- Risk Classification ----

    def classify_risk(self, tool_name: str, arguments: Dict[str, Any]) -> RiskClassification:
        """
        Deterministic risk classification.

        1. Start with base tool risk from config
        2. Check argument patterns for escalation
        3. Check rate-based escalation
        4. Return highest risk level found
        """
        reasons = []

        # 1. Base tool risk
        base_risk = self._tool_risk.get(tool_name, RiskLevel.MEDIUM)
        reasons.append(f"base_risk:{base_risk.value}")
        max_risk = base_risk

        # 2. Argument pattern escalation
        args_str = json.dumps(arguments, default=str).lower()
        for rule in self._compiled_rules:
            field_name = rule["field"]
            search_text = args_str
            if field_name != "*":
                search_text = str(arguments.get(field_name, "")).lower()

            if rule["regex"].search(search_text):
                escalate_to = rule["escalate_to"]
                if RISK_ORDER.get(escalate_to, 0) > RISK_ORDER.get(max_risk, 0):
                    max_risk = escalate_to
                    reasons.append(f"escalation:{rule['regex'].pattern}→{escalate_to.value}")

        # 3. Rate-based escalation
        rate_max = self._rate_escalation.get("max_actions_per_minute", 15)
        with self._rate_lock:
            now = time.time()
            cutoff = now - 60
            self._action_timestamps = [t for t in self._action_timestamps if t > cutoff]
            self._action_timestamps.append(now)
            if len(self._action_timestamps) > rate_max:
                rate_level = RiskLevel(self._rate_escalation.get("escalate_to", "medium"))
                if RISK_ORDER.get(rate_level, 0) > RISK_ORDER.get(max_risk, 0):
                    max_risk = rate_level
                    reasons.append(f"rate_escalation:{len(self._action_timestamps)}/min")

        # Determine approval requirements
        requires_approval = not self._auto_approve.get(max_risk.value, False)
        auto_secs = None
        if max_risk == RiskLevel.MEDIUM:
            auto_secs = self._auto_approve.get("medium_delay_secs", 0)

        return RiskClassification(
            level=max_risk,
            tool_name=tool_name,
            reasons=reasons,
            requires_approval=requires_approval,
            auto_approve_after_secs=auto_secs,
        )

    # ---- Approval Gating ----

    def request_approval(
        self,
        classification: RiskClassification,
        plan_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> GovernanceDecision:
        """
        Request approval for a classified action.

        LOW → auto-approved immediately.
        MEDIUM → auto-approved (with optional delay).
        HIGH/CRITICAL → returns "pending", requires human approval.
        """
        import uuid
        action_id = f"gov-{uuid.uuid4().hex[:8]}"

        decision = GovernanceDecision(
            action_id=action_id,
            classification=classification,
        )

        # Safe Degradation Mode (v8.5): override auto-approve under stress
        system_mode = self.get_system_mode()
        current_mode = system_mode.get("mode", "normal")

        if current_mode == "read_only" and classification.level != RiskLevel.LOW:
            # Read-only: only LOW auto-approved, everything else pending
            decision.status = "pending"
            classification.reasons.append(f"degradation:read_only")
            logger.warning(f"Safe Degradation: action {action_id} held (read_only mode)")
        elif current_mode == "conservative" and classification.level == RiskLevel.MEDIUM:
            # Conservative: MEDIUM no longer auto-approved
            decision.status = "pending"
            classification.reasons.append(f"degradation:conservative")
            logger.info(f"Safe Degradation: MEDIUM action {action_id} held (conservative mode)")
        elif not classification.requires_approval:
            decision.status = "approved"
            decision.approved_by = "auto"
            decision.approved_at = time.time()
        else:
            decision.status = "pending"

        # Persist
        self._store_decision(decision, plan_id, session_id)

        return decision

    def approve(self, action_id: str, approved_by: str = "user") -> bool:
        """Human approves a pending action."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "UPDATE governance_actions SET status = 'approved', approved_by = ?, approved_at = ? "
                "WHERE action_id = ? AND status = 'pending'",
                (approved_by, time.time(), action_id),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            cursor.close()

    def deny(self, action_id: str, reason: str = "") -> bool:
        """Human denies a pending action."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "UPDATE governance_actions SET status = 'denied', denied_reason = ? "
                "WHERE action_id = ? AND status = 'pending'",
                (reason, action_id),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            cursor.close()

    def check_constraints(
        self, tool_name: str, arguments: Dict[str, Any], constraints: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Verify execution stays within declared constraints."""
        # Currently constraints are informational — future: enforce file size limits, etc.
        return True, ""

    def record_execution(self, action_id: str, success: bool, output_preview: str = ""):
        """Record that an approved action was executed."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            result_json = json.dumps({"success": success, "preview": output_preview[:200]})
            status = "executed" if success else "failed"
            cursor.execute(
                "UPDATE governance_actions SET status = ?, executed_at = ?, result_json = ? "
                "WHERE action_id = ?",
                (status, time.time(), result_json, action_id),
            )
            conn.commit()
        finally:
            cursor.close()

    # ---- Queries ----

    def get_pending_approvals(self) -> List[Dict[str, Any]]:
        """Get all actions waiting for human approval."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT * FROM governance_actions WHERE status = 'pending' ORDER BY created_at DESC"
            )
            return [self._row_to_dict(row) for row in cursor.fetchall()]
        finally:
            cursor.close()

    def get_history(self, limit: int = 20, risk_level: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent governance decisions."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            if risk_level:
                cursor.execute(
                    "SELECT * FROM governance_actions WHERE risk_level = ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (risk_level, limit),
                )
            else:
                cursor.execute(
                    "SELECT * FROM governance_actions ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                )
            return [self._row_to_dict(row) for row in cursor.fetchall()]
        finally:
            cursor.close()

    def get_stats(self) -> Dict[str, Any]:
        """Governance statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) as cnt FROM governance_actions")
            total = cursor.fetchone()["cnt"]

            cursor.execute("SELECT COUNT(*) as cnt FROM governance_actions WHERE status = 'pending'")
            pending = cursor.fetchone()["cnt"]

            cursor.execute(
                "SELECT risk_level, COUNT(*) as cnt FROM governance_actions GROUP BY risk_level"
            )
            by_risk = {row["risk_level"]: row["cnt"] for row in cursor.fetchall()}

            cursor.execute(
                "SELECT status, COUNT(*) as cnt FROM governance_actions GROUP BY status"
            )
            by_status = {row["status"]: row["cnt"] for row in cursor.fetchall()}

            auto = by_status.get("executed", 0) + by_status.get("approved", 0)
            autonomy = round(auto / total, 2) if total > 0 else 0

            system_mode = self.get_system_mode()

            return {
                "enabled": True,
                "version": "8.5",
                "total_actions": total,
                "pending_approvals": pending,
                "by_risk_level": by_risk,
                "by_status": by_status,
                "autonomy_ratio": autonomy,
                "system_mode": system_mode.get("mode", "normal"),
                "system_mode_reasons": system_mode.get("reasons", []),
            }
        finally:
            cursor.close()

    # ---- Safe Degradation Mode (v8.5) ----

    def set_dashboard(self, dashboard) -> None:
        """Inject DashboardService for stress-aware degradation."""
        self._dashboard = dashboard

    def set_audit(self, audit) -> None:
        """Inject AuditLogger for integrity-aware degradation."""
        self._audit = audit

    def get_system_mode(self) -> Dict[str, Any]:
        """
        Determine current system operating mode based on stress indicators.

        Modes:
        - "normal": standard operation, all auto-approve rules apply
        - "conservative": elevated caution — MEDIUM no longer auto-approved
        - "read_only": only LOW actions auto-approved, everything else pending

        Checks (cached for 60s to avoid overhead):
        1. error_rate > 20% → conservative
        2. pending_approvals > 5 → conservative
        3. audit chain verification fails → read_only
        """
        now = time.time()
        if self._system_mode_cache and (now - self._system_mode_cache_ts) < self._system_mode_cache_ttl:
            return self._system_mode_cache

        mode = "normal"
        reasons = []

        # 1. Check pending approvals (from own data — no external dependency)
        try:
            pending = self.get_pending_approvals()
            if len(pending) > 5:
                mode = "conservative"
                reasons.append(f"pending_approvals={len(pending)}")
        except Exception:
            pass

        # 2. Check error rate from dashboard
        if self._dashboard and mode != "read_only":
            try:
                snapshot = self._dashboard.capture_snapshot()
                if snapshot.total_requests_24h > 0 and snapshot.error_rate > 0.20:
                    mode = "conservative"
                    reasons.append(f"error_rate={snapshot.error_rate:.0%}")
            except Exception:
                pass

        # 3. Check audit integrity (most severe → read_only)
        if self._audit:
            try:
                chain_result = self._audit.verify_chain()
                if not chain_result.valid and chain_result.entries_checked > 0:
                    mode = "read_only"
                    reasons.append(f"audit_chain_broken_at={chain_result.index}")
            except Exception:
                pass

        self._system_mode_cache = {"mode": mode, "reasons": reasons, "checked_at": now}
        self._system_mode_cache_ts = now
        logger.debug(f"System mode: {mode} ({', '.join(reasons) if reasons else 'all clear'})")
        return self._system_mode_cache

    # ---- Internal ----

    def _store_decision(
        self, decision: GovernanceDecision,
        plan_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO governance_actions "
                "(action_id, plan_id, tool_name, arguments_json, risk_level, risk_reasons, "
                "constraints_json, status, approved_by, created_at, approved_at, session_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    decision.action_id,
                    plan_id,
                    decision.classification.tool_name,
                    "{}",  # arguments stored separately for privacy
                    decision.classification.level.value,
                    json.dumps(decision.classification.reasons),
                    json.dumps(decision.constraints_applied),
                    decision.status,
                    decision.approved_by,
                    time.time(),
                    decision.approved_at,
                    session_id,
                ),
            )
            conn.commit()
        except Exception as e:
            logger.debug(f"Failed to store governance decision: {e}")
            conn.rollback()
        finally:
            cursor.close()

    @staticmethod
    def _row_to_dict(row) -> Dict[str, Any]:
        d = dict(row)
        for json_field in ("risk_reasons", "constraints_json", "result_json"):
            if d.get(json_field):
                try:
                    d[json_field] = json.loads(d[json_field])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d
