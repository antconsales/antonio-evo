"""
Dashboard Service (v8.5) — Zero-effort KPI visibility.

Aggregates metrics from all subsystems into a single view:
- Request volume and success rates
- Governance risk distribution and autonomy ratio
- Workflow plan status
- Memory usage
- Alert detection (threshold-based)

The user never asks "what happened?" — it's surfaced proactively.
"""

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class KPISnapshot:
    """Point-in-time KPI capture."""
    timestamp: float
    total_requests_24h: int = 0
    success_rate: float = 0.0
    avg_response_ms: int = 0
    governance_by_risk: Dict[str, int] = field(default_factory=dict)
    approval_rate: float = 0.0
    pending_approvals: int = 0
    autonomy_ratio: float = 0.0
    active_plans: int = 0
    completed_plans_24h: int = 0
    failed_plans_24h: int = 0
    error_rate: float = 0.0
    top_errors: List[str] = field(default_factory=list)
    memory_neurons: int = 0
    active_schedules: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "total_requests_24h": self.total_requests_24h,
            "success_rate": self.success_rate,
            "avg_response_ms": self.avg_response_ms,
            "governance_by_risk": self.governance_by_risk,
            "approval_rate": self.approval_rate,
            "pending_approvals": self.pending_approvals,
            "autonomy_ratio": self.autonomy_ratio,
            "active_plans": self.active_plans,
            "completed_plans_24h": self.completed_plans_24h,
            "failed_plans_24h": self.failed_plans_24h,
            "error_rate": self.error_rate,
            "top_errors": self.top_errors,
            "memory_neurons": self.memory_neurons,
            "active_schedules": self.active_schedules,
        }


class DashboardService:
    """
    Aggregated KPI dashboard for all Antonio subsystems.

    Collects stats from: governance, workflow, audit, self-improvement,
    memory_storage. Provides natural language briefings and alert detection.
    """

    ALERT_THRESHOLDS = {
        "success_rate_min": 0.80,
        "pending_approvals_max": 5,
        "error_rate_max": 0.20,
        "failed_plans_max": 3,
    }

    def __init__(
        self,
        db_path: str = "data/evomemory.db",
        governance=None,
        workflow=None,
        audit=None,
        self_improvement=None,
        memory_storage=None,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()

        self._governance = governance
        self._workflow = workflow
        self._audit = audit
        self._self_improvement = self_improvement
        self._memory_storage = memory_storage

        self._init_schema()
        logger.info("Dashboard Service (v8.5) initialized")

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
                CREATE TABLE IF NOT EXISTS kpi_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    autonomy_ratio REAL DEFAULT 0,
                    success_rate REAL DEFAULT 0,
                    total_requests INTEGER DEFAULT 0,
                    pending_approvals INTEGER DEFAULT 0
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_kpi_ts
                ON kpi_snapshots(timestamp DESC)
            """)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    def capture_snapshot(self) -> KPISnapshot:
        """
        Capture current KPI snapshot by aggregating all subsystems.

        Returns:
            KPISnapshot with latest metrics
        """
        now = time.time()
        snapshot = KPISnapshot(timestamp=now)

        # Audit stats (requests, success rate, response time)
        if self._audit:
            try:
                cutoff_24h = now - 86400
                entries = self._audit.get_all()
                recent = [e for e in entries if e.timestamp >= cutoff_24h
                          and e.event_type == "request_cycle"]

                snapshot.total_requests_24h = len(recent)
                if recent:
                    successes = sum(
                        1 for e in recent
                        if e.payload.get("response", {}).get("success", False)
                    )
                    snapshot.success_rate = round(successes / len(recent), 2)
                    snapshot.error_rate = round(1 - snapshot.success_rate, 2)

                    elapsed_vals = [
                        e.payload.get("elapsed_ms", 0) for e in recent
                        if e.payload.get("elapsed_ms", 0) > 0
                    ]
                    if elapsed_vals:
                        snapshot.avg_response_ms = int(sum(elapsed_vals) / len(elapsed_vals))

                    # Top errors
                    errors = [
                        e.payload.get("response", {}).get("error", "unknown")
                        for e in recent
                        if not e.payload.get("response", {}).get("success", False)
                        and e.payload.get("response", {}).get("error")
                    ]
                    from collections import Counter
                    snapshot.top_errors = [err for err, _ in Counter(errors).most_common(5)]
            except Exception as e:
                logger.debug(f"Dashboard: audit stats failed: {e}")

        # Governance stats
        if self._governance:
            try:
                gov_stats = self._governance.get_stats()
                snapshot.governance_by_risk = gov_stats.get("by_risk_level", {})
                snapshot.pending_approvals = gov_stats.get("pending_approvals", 0)
                snapshot.autonomy_ratio = gov_stats.get("autonomy_ratio", 0)

                by_status = gov_stats.get("by_status", {})
                total_gov = gov_stats.get("total_actions", 0)
                if total_gov > 0:
                    approved = by_status.get("approved", 0) + by_status.get("executed", 0)
                    snapshot.approval_rate = round(approved / total_gov, 2)
            except Exception as e:
                logger.debug(f"Dashboard: governance stats failed: {e}")

        # Workflow stats
        if self._workflow:
            try:
                wf_stats = self._workflow.get_stats()
                cutoff_24h = now - 86400

                # Active plans (running or approved)
                running = self._workflow.list_plans(status="running")
                approved = self._workflow.list_plans(status="approved")
                snapshot.active_plans = len(running) + len(approved)

                # Completed/failed in 24h
                completed = self._workflow.list_plans(status="completed", limit=50)
                snapshot.completed_plans_24h = sum(
                    1 for p in completed
                    if p.completed_at and p.completed_at >= cutoff_24h
                )
                failed = self._workflow.list_plans(status="failed", limit=50)
                snapshot.failed_plans_24h = sum(
                    1 for p in failed
                    if p.completed_at and p.completed_at >= cutoff_24h
                )

                snapshot.active_schedules = wf_stats.get("active_schedules", 0)
            except Exception as e:
                logger.debug(f"Dashboard: workflow stats failed: {e}")

        # Memory stats
        if self._memory_storage:
            try:
                mem_stats = self._memory_storage.get_stats()
                snapshot.memory_neurons = mem_stats.get("total_neurons", 0)
            except Exception as e:
                logger.debug(f"Dashboard: memory stats failed: {e}")

        # Persist snapshot
        self._store_snapshot(snapshot)

        return snapshot

    def get_briefing(self, style: str = "concise") -> str:
        """
        Natural language briefing of current system status.

        Args:
            style: "concise" (2-3 sentences) or "detailed" (full breakdown)
        """
        snapshot = self.capture_snapshot()

        if style == "concise":
            parts = []
            parts.append(
                f"Ultime 24h: {snapshot.total_requests_24h} richieste, "
                f"{snapshot.success_rate * 100:.0f}% successo"
                f"{f', {snapshot.avg_response_ms}ms media' if snapshot.avg_response_ms else ''}."
            )
            if snapshot.pending_approvals > 0:
                parts.append(f"{snapshot.pending_approvals} azioni in attesa di approvazione.")
            parts.append(
                f"Autonomia: {snapshot.autonomy_ratio * 100:.0f}% "
                f"({snapshot.memory_neurons} neuroni in memoria)."
            )
            return " ".join(parts)

        # Detailed briefing
        lines = ["## Dashboard KPI", ""]
        lines.append(f"**Periodo**: ultime 24 ore")
        lines.append(f"**Richieste**: {snapshot.total_requests_24h}")
        lines.append(f"**Successo**: {snapshot.success_rate * 100:.0f}%")
        lines.append(f"**Tempo medio**: {snapshot.avg_response_ms}ms")
        lines.append("")
        lines.append("### Governance")
        lines.append(f"- Autonomia: {snapshot.autonomy_ratio * 100:.0f}%")
        lines.append(f"- Approvazioni pending: {snapshot.pending_approvals}")
        lines.append(f"- Distribuzione rischio: {snapshot.governance_by_risk}")
        lines.append("")
        lines.append("### Workflow")
        lines.append(f"- Piani attivi: {snapshot.active_plans}")
        lines.append(f"- Completati (24h): {snapshot.completed_plans_24h}")
        lines.append(f"- Falliti (24h): {snapshot.failed_plans_24h}")
        lines.append(f"- Schedules attive: {snapshot.active_schedules}")
        lines.append("")
        lines.append("### Memoria")
        lines.append(f"- Neuroni: {snapshot.memory_neurons}")

        if snapshot.top_errors:
            lines.append("")
            lines.append("### Errori principali")
            for err in snapshot.top_errors[:3]:
                lines.append(f"- {err}")

        # Alerts
        alerts = self.check_alerts(snapshot)
        if alerts:
            lines.append("")
            lines.append("### Avvisi")
            for alert in alerts:
                lines.append(f"- {alert['message']}")

        return "\n".join(lines)

    def get_autonomy_report(self) -> Dict[str, Any]:
        """70/30 split analysis — how autonomous is the agent?"""
        if not self._governance:
            return {"error": "Governance engine not available"}

        try:
            stats = self._governance.get_stats()
            total = stats.get("total_actions", 0)
            by_status = stats.get("by_status", {})

            auto_approved = by_status.get("executed", 0) + by_status.get("approved", 0)
            human_approved = sum(
                1 for _ in []  # Count from history where approved_by = 'user'
            )
            denied = by_status.get("denied", 0)

            # Better: query governance history for human approvals
            history = self._governance.get_history(limit=1000)
            human_count = sum(1 for h in history if h.get("approved_by") == "user")

            return {
                "total_actions": total,
                "auto_approved": auto_approved - human_count,
                "human_approved": human_count,
                "denied": denied,
                "pending": stats.get("pending_approvals", 0),
                "autonomy_ratio": stats.get("autonomy_ratio", 0),
                "target_ratio": 0.70,
                "on_target": stats.get("autonomy_ratio", 0) >= 0.70,
                "by_risk_level": stats.get("by_risk_level", {}),
            }
        except Exception as e:
            return {"error": str(e)}

    def get_trend(self, metric: str = "success_rate", days: int = 7) -> List[Dict[str, Any]]:
        """Historical data points for a metric."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cutoff = time.time() - (days * 86400)
            cursor.execute(
                "SELECT timestamp, snapshot_json FROM kpi_snapshots "
                "WHERE timestamp >= ? ORDER BY timestamp ASC",
                (cutoff,),
            )
            points = []
            for row in cursor.fetchall():
                try:
                    data = json.loads(row["snapshot_json"])
                    points.append({
                        "timestamp": row["timestamp"],
                        "value": data.get(metric, 0),
                    })
                except (json.JSONDecodeError, TypeError):
                    continue
            return points
        finally:
            cursor.close()

    def check_alerts(self, snapshot: KPISnapshot = None) -> List[Dict[str, Any]]:
        """Threshold-based anomaly detection."""
        if snapshot is None:
            snapshot = self.capture_snapshot()

        alerts = []

        if snapshot.total_requests_24h > 0:
            if snapshot.success_rate < self.ALERT_THRESHOLDS["success_rate_min"]:
                alerts.append({
                    "type": "success_rate_low",
                    "severity": "high",
                    "message": f"Tasso di successo basso: {snapshot.success_rate * 100:.0f}% "
                               f"(soglia: {self.ALERT_THRESHOLDS['success_rate_min'] * 100:.0f}%)",
                    "value": snapshot.success_rate,
                    "threshold": self.ALERT_THRESHOLDS["success_rate_min"],
                })

            if snapshot.error_rate > self.ALERT_THRESHOLDS["error_rate_max"]:
                alerts.append({
                    "type": "error_rate_high",
                    "severity": "high",
                    "message": f"Tasso di errore alto: {snapshot.error_rate * 100:.0f}% "
                               f"(soglia: {self.ALERT_THRESHOLDS['error_rate_max'] * 100:.0f}%)",
                    "value": snapshot.error_rate,
                    "threshold": self.ALERT_THRESHOLDS["error_rate_max"],
                })

        if snapshot.pending_approvals > self.ALERT_THRESHOLDS["pending_approvals_max"]:
            alerts.append({
                "type": "pending_approvals_high",
                "severity": "medium",
                "message": f"{snapshot.pending_approvals} azioni in attesa di approvazione "
                           f"(soglia: {self.ALERT_THRESHOLDS['pending_approvals_max']})",
                "value": snapshot.pending_approvals,
                "threshold": self.ALERT_THRESHOLDS["pending_approvals_max"],
            })

        if snapshot.failed_plans_24h > self.ALERT_THRESHOLDS["failed_plans_max"]:
            alerts.append({
                "type": "failed_plans_high",
                "severity": "medium",
                "message": f"{snapshot.failed_plans_24h} piani falliti nelle ultime 24h "
                           f"(soglia: {self.ALERT_THRESHOLDS['failed_plans_max']})",
                "value": snapshot.failed_plans_24h,
                "threshold": self.ALERT_THRESHOLDS["failed_plans_max"],
            })

        return alerts

    def get_stats(self) -> Dict[str, Any]:
        """Dashboard service stats."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) as cnt FROM kpi_snapshots")
            total = cursor.fetchone()["cnt"]
            return {
                "enabled": True,
                "version": "8.5",
                "total_snapshots": total,
                "has_governance": self._governance is not None,
                "has_workflow": self._workflow is not None,
                "has_audit": self._audit is not None,
            }
        finally:
            cursor.close()

    # ---- Internal ----

    def _store_snapshot(self, snapshot: KPISnapshot):
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO kpi_snapshots "
                "(timestamp, snapshot_json, autonomy_ratio, success_rate, total_requests, pending_approvals) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    snapshot.timestamp,
                    json.dumps(snapshot.to_dict()),
                    snapshot.autonomy_ratio,
                    snapshot.success_rate,
                    snapshot.total_requests_24h,
                    snapshot.pending_approvals,
                ),
            )
            conn.commit()
        except Exception as e:
            logger.debug(f"Failed to store KPI snapshot: {e}")
            conn.rollback()
        finally:
            cursor.close()
