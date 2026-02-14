"""
Workflow Orchestrator (v8.5) — Multi-step action plans with retries and timeouts.

Provides:
- ActionPlan creation with sequential steps
- Per-step timeout enforcement and configurable retries
- User approval before execution
- Step-by-step execution via ToolExecutor
- Scheduled task support (daily, hourly, weekly)
"""

import json
import logging
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ActionStep:
    """A single step in an action plan."""
    step_number: int
    tool_name: str
    arguments: Dict[str, Any]
    description: str = ""
    depends_on: Optional[int] = None
    status: str = "pending"  # pending, running, completed, failed, skipped
    result: Optional[str] = None
    timeout_secs: int = 60
    max_retries: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "description": self.description,
            "depends_on": self.depends_on,
            "status": self.status,
            "result": self.result[:200] if self.result else None,
            "timeout_secs": self.timeout_secs,
            "max_retries": self.max_retries,
        }


@dataclass
class ActionPlan:
    """A multi-step action plan."""
    id: str
    title: str
    steps: List[ActionStep]
    status: str = "draft"  # draft, approved, running, completed, failed
    created_at: float = 0.0
    approved_at: Optional[float] = None
    completed_at: Optional[float] = None
    result_summary: Optional[str] = None
    session_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "steps": [s.to_dict() for s in self.steps],
            "status": self.status,
            "created_at": self.created_at,
            "approved_at": self.approved_at,
            "completed_at": self.completed_at,
            "result_summary": self.result_summary,
            "session_id": self.session_id,
        }


class WorkflowOrchestrator:
    """
    Manages action plans and scheduled tasks.

    SQLite tables:
    - action_plans: id, title, steps_json, status, timestamps
    - scheduled_tasks: name, cron, tool, arguments, enabled
    """

    def __init__(self, db_path: str = "data/evomemory.db", tool_executor=None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.tool_executor = tool_executor
        self._local = threading.local()
        self._init_schema()

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
                CREATE TABLE IF NOT EXISTS action_plans (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    steps_json TEXT NOT NULL DEFAULT '[]',
                    status TEXT DEFAULT 'draft',
                    created_at REAL NOT NULL,
                    approved_at REAL,
                    completed_at REAL,
                    result_summary TEXT,
                    session_id TEXT
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ap_status
                ON action_plans(status)
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scheduled_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    cron_expression TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    tool_arguments TEXT DEFAULT '{}',
                    enabled BOOLEAN DEFAULT 1,
                    last_run REAL,
                    next_run REAL,
                    created_at REAL NOT NULL
                )
            """)

            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    # ---- Action Plans ----

    def create_plan(
        self, title: str, steps: List[Dict[str, Any]], session_id: Optional[str] = None
    ) -> ActionPlan:
        """Create a new action plan (status=draft)."""
        import uuid

        plan_id = f"plan-{uuid.uuid4().hex[:8]}"
        now = time.time()

        action_steps = []
        for i, s in enumerate(steps):
            action_steps.append(ActionStep(
                step_number=i + 1,
                tool_name=s.get("tool", s.get("tool_name", "")),
                arguments=s.get("arguments", {}),
                description=s.get("description", ""),
                depends_on=s.get("depends_on"),
                timeout_secs=s.get("timeout_secs", 60),
                max_retries=s.get("max_retries", 1),
            ))

        plan = ActionPlan(
            id=plan_id,
            title=title,
            steps=action_steps,
            status="draft",
            created_at=now,
            session_id=session_id,
        )

        # Store in DB
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            steps_json = json.dumps([s.to_dict() for s in action_steps])
            cursor.execute(
                "INSERT INTO action_plans (id, title, steps_json, status, created_at, session_id) "
                "VALUES (?, ?, ?, 'draft', ?, ?)",
                (plan_id, title, steps_json, now, session_id),
            )
            conn.commit()
        finally:
            cursor.close()

        return plan

    def approve_plan(self, plan_id: str) -> bool:
        """Mark a plan as approved (ready for execution)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "UPDATE action_plans SET status = 'approved', approved_at = ? WHERE id = ? AND status = 'draft'",
                (time.time(), plan_id),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            cursor.close()

    def execute_plan(self, plan_id: str, callback: Optional[Callable] = None) -> ActionPlan:
        """
        Execute an approved plan step-by-step.

        Args:
            plan_id: Plan to execute
            callback: Optional callback(step_number, status, result) per step

        Returns updated ActionPlan.
        """
        plan = self.get_plan(plan_id)
        if not plan or plan.status != "approved":
            raise ValueError(f"Plan {plan_id} not found or not approved")

        if not self.tool_executor:
            raise ValueError("No tool executor configured")

        # Mark as running
        self._update_plan_status(plan_id, "running")
        plan.status = "running"

        results = []
        all_ok = True
        _executor_pool = ThreadPoolExecutor(max_workers=1)

        for step in plan.steps:
            # Check dependency
            if step.depends_on:
                dep_step = next((s for s in plan.steps if s.step_number == step.depends_on), None)
                if dep_step and dep_step.status == "failed":
                    step.status = "skipped"
                    step.result = "Skipped: dependency failed"
                    continue

            step.status = "running"
            if callback:
                try:
                    callback(step.step_number, "running", None)
                except Exception:
                    pass

            # Retry loop with timeout (v8.5)
            attempts = max(step.max_retries, 1)
            for attempt in range(1, attempts + 1):
                try:
                    future = _executor_pool.submit(
                        self.tool_executor.execute, step.tool_name, step.arguments
                    )
                    tool_result = future.result(timeout=step.timeout_secs)

                    step.result = tool_result.output[:1000] if tool_result.output else ""
                    step.status = "completed" if tool_result.success else "failed"
                    if tool_result.success:
                        break  # Success — exit retry loop
                    if attempt < attempts:
                        logger.info(f"Step {step.step_number} failed (attempt {attempt}/{attempts}), retrying...")
                        continue
                except FuturesTimeout:
                    step.status = "failed"
                    step.result = f"Timeout after {step.timeout_secs}s (attempt {attempt}/{attempts})"
                    logger.warning(f"Step {step.step_number} timed out after {step.timeout_secs}s")
                    if attempt < attempts:
                        continue
                except Exception as e:
                    step.status = "failed"
                    step.result = str(e)[:500]
                    if attempt < attempts:
                        continue

            if step.status != "completed":
                all_ok = False
            results.append(f"Step {step.step_number}: {step.status}")

            if callback:
                try:
                    callback(step.step_number, step.status, step.result)
                except Exception:
                    pass

        _executor_pool.shutdown(wait=False)

        # Finalize
        plan.status = "completed" if all_ok else "failed"
        plan.completed_at = time.time()
        plan.result_summary = "; ".join(results)

        # Persist
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            steps_json = json.dumps([s.to_dict() for s in plan.steps])
            cursor.execute(
                "UPDATE action_plans SET steps_json = ?, status = ?, completed_at = ?, result_summary = ? WHERE id = ?",
                (steps_json, plan.status, plan.completed_at, plan.result_summary, plan_id),
            )
            conn.commit()
        finally:
            cursor.close()

        return plan

    def get_plan(self, plan_id: str) -> Optional[ActionPlan]:
        """Get a plan by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM action_plans WHERE id = ?", (plan_id,))
            row = cursor.fetchone()
            return self._row_to_plan(row) if row else None
        finally:
            cursor.close()

    def list_plans(self, status: Optional[str] = None, limit: int = 20) -> List[ActionPlan]:
        """List plans, optionally filtered by status."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            if status:
                cursor.execute(
                    "SELECT * FROM action_plans WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status, limit),
                )
            else:
                cursor.execute(
                    "SELECT * FROM action_plans ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                )
            return [self._row_to_plan(row) for row in cursor.fetchall()]
        finally:
            cursor.close()

    def _update_plan_status(self, plan_id: str, status: str):
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "UPDATE action_plans SET status = ? WHERE id = ?",
                (status, plan_id),
            )
            conn.commit()
        finally:
            cursor.close()

    # ---- Scheduled Tasks ----

    def schedule_task(
        self,
        name: str,
        cron_expression: str,
        tool_name: str,
        tool_arguments: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Schedule a recurring task.

        cron_expression: "daily 09:00", "hourly", "weekly mon"
        """
        now = time.time()
        next_run = self._compute_next_run(cron_expression, now)

        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT OR REPLACE INTO scheduled_tasks (name, cron_expression, tool_name, tool_arguments, enabled, next_run, created_at) "
                "VALUES (?, ?, ?, ?, 1, ?, ?)",
                (name, cron_expression, tool_name, json.dumps(tool_arguments or {}), next_run, now),
            )
            conn.commit()
            return {
                "name": name,
                "cron": cron_expression,
                "tool": tool_name,
                "next_run": next_run,
            }
        finally:
            cursor.close()

    def get_pending_schedules(self) -> List[Dict[str, Any]]:
        """Get tasks due for execution."""
        now = time.time()
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT * FROM scheduled_tasks WHERE enabled = 1 AND (next_run IS NULL OR next_run <= ?)",
                (now,),
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            cursor.close()

    def get_schedules(self) -> List[Dict[str, Any]]:
        """Get all scheduled tasks."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM scheduled_tasks ORDER BY name")
            return [dict(row) for row in cursor.fetchall()]
        finally:
            cursor.close()

    def delete_schedule(self, task_id: int) -> bool:
        """Delete a scheduled task."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM scheduled_tasks WHERE id = ?", (task_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            cursor.close()

    def run_scheduler_tick(self) -> List[Dict[str, Any]]:
        """Check and execute due scheduled tasks. Called periodically."""
        if not self.tool_executor:
            return []

        pending = self.get_pending_schedules()
        results = []

        for task in pending:
            try:
                args = json.loads(task.get("tool_arguments", "{}"))
                result = self.tool_executor.execute(task["tool_name"], args)

                now = time.time()
                next_run = self._compute_next_run(task["cron_expression"], now)

                conn = self._get_connection()
                cursor = conn.cursor()
                try:
                    cursor.execute(
                        "UPDATE scheduled_tasks SET last_run = ?, next_run = ? WHERE id = ?",
                        (now, next_run, task["id"]),
                    )
                    conn.commit()
                finally:
                    cursor.close()

                results.append({
                    "task": task["name"],
                    "success": result.success,
                    "output": result.output[:200] if result.output else "",
                })
            except Exception as e:
                results.append({
                    "task": task.get("name", "?"),
                    "success": False,
                    "error": str(e),
                })

        return results

    # ---- Stats ----

    def get_stats(self) -> Dict[str, Any]:
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) as cnt FROM action_plans")
            total_plans = cursor.fetchone()["cnt"]

            cursor.execute("SELECT COUNT(*) as cnt FROM action_plans WHERE status = 'completed'")
            completed = cursor.fetchone()["cnt"]

            cursor.execute("SELECT COUNT(*) as cnt FROM scheduled_tasks WHERE enabled = 1")
            active_schedules = cursor.fetchone()["cnt"]

            return {
                "enabled": True,
                "version": "8.0",
                "total_plans": total_plans,
                "completed_plans": completed,
                "active_schedules": active_schedules,
            }
        finally:
            cursor.close()

    # ---- Helpers ----

    def _row_to_plan(self, row) -> ActionPlan:
        steps_data = json.loads(row["steps_json"]) if row["steps_json"] else []
        steps = [
            ActionStep(
                step_number=s.get("step_number", 0),
                tool_name=s.get("tool_name", ""),
                arguments=s.get("arguments", {}),
                description=s.get("description", ""),
                depends_on=s.get("depends_on"),
                status=s.get("status", "pending"),
                result=s.get("result"),
                timeout_secs=s.get("timeout_secs", 60),
                max_retries=s.get("max_retries", 1),
            )
            for s in steps_data
        ]
        return ActionPlan(
            id=row["id"],
            title=row["title"],
            steps=steps,
            status=row["status"],
            created_at=row["created_at"],
            approved_at=row["approved_at"],
            completed_at=row["completed_at"],
            result_summary=row["result_summary"],
            session_id=row["session_id"],
        )

    @staticmethod
    def _compute_next_run(cron_expr: str, from_time: float) -> float:
        """Compute next run time from a simple cron expression."""
        cron = cron_expr.lower().strip()

        if cron == "hourly":
            return from_time + 3600
        elif cron.startswith("daily"):
            return from_time + 86400
        elif cron.startswith("weekly"):
            return from_time + 604800
        elif "min" in cron:
            # e.g. "every 30min"
            try:
                minutes = int("".join(c for c in cron if c.isdigit()))
                return from_time + (minutes * 60)
            except ValueError:
                return from_time + 3600

        # Default: next hour
        return from_time + 3600
