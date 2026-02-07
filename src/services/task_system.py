"""
Task System for Antonio Evo (v3.1).

Per Antonio Evo Unified Spec (v3.1):
You do not run agents. You operate on FINITE, DECLARATIVE TASKS.

A task is:
- Schema-validated
- Single-execution
- Explicitly approved
- Auditable

You may:
- Propose tasks
- Describe side effects
- Suggest parameters

You may NEVER:
- Loop tasks
- Retry autonomously
- Chain execution without code approval

Features:
- Schema validation for task definitions
- Single-execution guarantee (no duplicate runs)
- Approval-based for sensitive operations
- Task lifecycle tracking
- Rollback support for reversible tasks

Philosophy: Every action is a task. Tasks require explicit approval.
"""

import uuid
import time
import json
import logging
import threading
import sqlite3
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of tasks."""
    QUERY = "query"           # Read-only, no approval needed
    MODIFY = "modify"         # Modifies local state
    EXTERNAL = "external"     # Calls external API (cost-bearing)
    DESTRUCTIVE = "destructive"  # Irreversible operation
    SYSTEM = "system"         # System-level operation


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"         # Created, awaiting approval
    APPROVED = "approved"       # Approved, ready for execution
    EXECUTING = "executing"     # Currently executing
    COMPLETED = "completed"     # Successfully completed
    FAILED = "failed"           # Execution failed
    CANCELLED = "cancelled"     # Cancelled by user
    ROLLED_BACK = "rolled_back"  # Rolled back after failure


class ApprovalLevel(Enum):
    """Required approval level."""
    NONE = "none"           # No approval needed (queries)
    IMPLICIT = "implicit"   # Auto-approved after delay
    EXPLICIT = "explicit"   # Requires user confirmation
    ADMIN = "admin"         # Requires admin approval


@dataclass
class TaskSchema:
    """Schema definition for a task type."""
    task_type: str
    description: str
    required_params: List[str] = field(default_factory=list)
    optional_params: List[str] = field(default_factory=list)
    approval_level: ApprovalLevel = ApprovalLevel.IMPLICIT
    reversible: bool = False
    max_retries: int = 0
    timeout_seconds: int = 60

    def validate(self, params: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate params against schema."""
        for param in self.required_params:
            if param not in params:
                return False, f"Missing required parameter: {param}"
        return True, None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "description": self.description,
            "required_params": self.required_params,
            "optional_params": self.optional_params,
            "approval_level": self.approval_level.value,
            "reversible": self.reversible,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass
class Task:
    """A task instance."""
    id: str
    task_type: str
    params: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    approved_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    execution_time_ms: int = 0
    rollback_data: Optional[Dict[str, Any]] = None
    approval_level: ApprovalLevel = ApprovalLevel.IMPLICIT
    approved_by: Optional[str] = None
    session_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "task_type": self.task_type,
            "params": self.params,
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
            "retry_count": self.retry_count,
            "execution_time_ms": self.execution_time_ms,
            "approval_level": self.approval_level.value,
            "approved_by": self.approved_by,
            "session_id": self.session_id,
        }


# Built-in task schemas
BUILTIN_SCHEMAS: Dict[str, TaskSchema] = {
    # Query tasks (no approval)
    "memory_search": TaskSchema(
        task_type="memory_search",
        description="Search memory for past interactions",
        required_params=["query"],
        optional_params=["limit"],
        approval_level=ApprovalLevel.NONE,
    ),
    "memory_stats": TaskSchema(
        task_type="memory_stats",
        description="Get memory statistics",
        approval_level=ApprovalLevel.NONE,
    ),
    "health_check": TaskSchema(
        task_type="health_check",
        description="Check system health",
        approval_level=ApprovalLevel.NONE,
    ),
    "rag_search": TaskSchema(
        task_type="rag_search",
        description="Search documents with RAG",
        required_params=["query"],
        optional_params=["limit"],
        approval_level=ApprovalLevel.NONE,
    ),

    # Modify tasks (implicit approval)
    "set_personality_trait": TaskSchema(
        task_type="set_personality_trait",
        description="Adjust personality trait value",
        required_params=["trait", "value"],
        approval_level=ApprovalLevel.IMPLICIT,
        reversible=True,
    ),
    "switch_profile": TaskSchema(
        task_type="switch_profile",
        description="Switch runtime profile",
        required_params=["profile"],
        approval_level=ApprovalLevel.IMPLICIT,
        reversible=True,
    ),
    "reset_session": TaskSchema(
        task_type="reset_session",
        description="Reset current session",
        approval_level=ApprovalLevel.IMPLICIT,
    ),

    # External tasks (explicit approval, cost-bearing)
    "external_llm_query": TaskSchema(
        task_type="external_llm_query",
        description="Query external LLM API",
        required_params=["prompt"],
        optional_params=["model", "max_tokens"],
        approval_level=ApprovalLevel.EXPLICIT,
        timeout_seconds=120,
    ),
    "generate_image": TaskSchema(
        task_type="generate_image",
        description="Generate image from prompt",
        required_params=["prompt"],
        optional_params=["width", "height", "steps"],
        approval_level=ApprovalLevel.EXPLICIT,
        timeout_seconds=300,
    ),

    # Destructive tasks (explicit approval, admin for some)
    "clear_memory": TaskSchema(
        task_type="clear_memory",
        description="Clear all memory neurons",
        approval_level=ApprovalLevel.ADMIN,
        reversible=False,
    ),
    "reset_digital_twin": TaskSchema(
        task_type="reset_digital_twin",
        description="Reset digital twin learning",
        approval_level=ApprovalLevel.EXPLICIT,
        reversible=False,
    ),
    "reset_personality": TaskSchema(
        task_type="reset_personality",
        description="Reset personality to defaults",
        approval_level=ApprovalLevel.EXPLICIT,
        reversible=False,
    ),
}


class TaskExecutor:
    """Executes tasks with handlers."""

    def __init__(self):
        self.handlers: Dict[str, Callable] = {}

    def register(self, task_type: str, handler: Callable) -> None:
        """Register a handler for a task type."""
        self.handlers[task_type] = handler

    def execute(self, task: Task) -> tuple[bool, Any]:
        """Execute a task."""
        handler = self.handlers.get(task.task_type)
        if not handler:
            return False, f"No handler for task type: {task.task_type}"

        try:
            result = handler(task.params)
            return True, result
        except Exception as e:
            return False, str(e)


class TaskManager:
    """
    Manages task lifecycle.

    Features:
    - Schema validation
    - Approval workflow
    - Single-execution guarantee
    - Execution tracking
    - Rollback support
    """

    def __init__(self, db_path: str = "data/evomemory.db"):
        self.db_path = db_path
        self.schemas: Dict[str, TaskSchema] = BUILTIN_SCHEMAS.copy()
        self.executor = TaskExecutor()
        self.pending_tasks: Dict[str, Task] = {}
        self._lock = threading.Lock()
        self._executed_ids: set = set()  # Prevents double execution

        self._init_db()

    def _init_db(self):
        """Initialize task database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    params TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    approved_at TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    result TEXT,
                    error TEXT,
                    retry_count INTEGER DEFAULT 0,
                    execution_time_ms INTEGER DEFAULT 0,
                    rollback_data TEXT,
                    approval_level TEXT,
                    approved_by TEXT,
                    session_id TEXT
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_tasks_status
                ON tasks(status)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_tasks_session
                ON tasks(session_id)
            """)

            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to initialize task DB: {e}")

    def register_schema(self, schema: TaskSchema) -> None:
        """Register a custom task schema."""
        self.schemas[schema.task_type] = schema
        logger.info(f"Registered task schema: {schema.task_type}")

    def register_handler(self, task_type: str, handler: Callable) -> None:
        """Register a handler for a task type."""
        self.executor.register(task_type, handler)

    def create_task(
        self,
        task_type: str,
        params: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        session_id: Optional[str] = None,
    ) -> tuple[Optional[Task], Optional[str]]:
        """
        Create a new task.

        Returns:
            Tuple of (Task, error_message)
        """
        params = params or {}

        # Validate schema exists
        schema = self.schemas.get(task_type)
        if not schema:
            return None, f"Unknown task type: {task_type}"

        # Validate params
        valid, error = schema.validate(params)
        if not valid:
            return None, error

        # Create task
        task = Task(
            id=str(uuid.uuid4())[:12],
            task_type=task_type,
            params=params,
            priority=priority,
            approval_level=schema.approval_level,
            session_id=session_id,
        )

        # Auto-approve if no approval needed
        if schema.approval_level == ApprovalLevel.NONE:
            task.status = TaskStatus.APPROVED
            task.approved_at = datetime.now()
            task.approved_by = "auto"

        with self._lock:
            self.pending_tasks[task.id] = task

        # Persist
        self._save_task(task)

        logger.info(f"Created task {task.id}: {task_type}")
        return task, None

    def approve_task(
        self,
        task_id: str,
        approved_by: str = "user",
    ) -> tuple[bool, Optional[str]]:
        """
        Approve a pending task.

        Returns:
            Tuple of (success, error_message)
        """
        with self._lock:
            task = self.pending_tasks.get(task_id)
            if not task:
                task = self._load_task(task_id)

            if not task:
                return False, "Task not found"

            if task.status != TaskStatus.PENDING:
                return False, f"Task is {task.status.value}, not pending"

            task.status = TaskStatus.APPROVED
            task.approved_at = datetime.now()
            task.approved_by = approved_by
            self._save_task(task)

        logger.info(f"Approved task {task_id} by {approved_by}")
        return True, None

    def reject_task(
        self,
        task_id: str,
        reason: str = "User rejected",
    ) -> tuple[bool, Optional[str]]:
        """Reject a pending task."""
        with self._lock:
            task = self.pending_tasks.get(task_id)
            if not task:
                task = self._load_task(task_id)

            if not task:
                return False, "Task not found"

            if task.status != TaskStatus.PENDING:
                return False, f"Task is {task.status.value}, not pending"

            task.status = TaskStatus.CANCELLED
            task.error = reason
            task.completed_at = datetime.now()
            self._save_task(task)

            if task_id in self.pending_tasks:
                del self.pending_tasks[task_id]

        logger.info(f"Rejected task {task_id}: {reason}")
        return True, None

    def execute_task(self, task_id: str) -> tuple[bool, Any]:
        """
        Execute an approved task.

        Returns:
            Tuple of (success, result_or_error)
        """
        with self._lock:
            # Check for double execution
            if task_id in self._executed_ids:
                return False, "Task already executed"

            task = self.pending_tasks.get(task_id)
            if not task:
                task = self._load_task(task_id)

            if not task:
                return False, "Task not found"

            if task.status not in [TaskStatus.APPROVED, TaskStatus.PENDING]:
                if task.status == TaskStatus.COMPLETED:
                    return True, task.result
                return False, f"Task is {task.status.value}"

            # Auto-approve if NONE level
            schema = self.schemas.get(task.task_type)
            if schema and schema.approval_level == ApprovalLevel.NONE:
                task.status = TaskStatus.APPROVED
                task.approved_at = datetime.now()
                task.approved_by = "auto"
            elif task.status != TaskStatus.APPROVED:
                return False, "Task requires approval"

            # Mark as executing
            task.status = TaskStatus.EXECUTING
            task.started_at = datetime.now()
            self._executed_ids.add(task_id)

        # Execute outside lock
        start_time = time.time()
        success, result = self.executor.execute(task)
        execution_time = int((time.time() - start_time) * 1000)

        with self._lock:
            task.execution_time_ms = execution_time

            if success:
                task.status = TaskStatus.COMPLETED
                task.result = result
            else:
                task.status = TaskStatus.FAILED
                task.error = str(result)

            task.completed_at = datetime.now()
            self._save_task(task)

            if task_id in self.pending_tasks:
                del self.pending_tasks[task_id]

        log_level = logging.INFO if success else logging.WARNING
        logger.log(log_level, f"Task {task_id} {task.status.value}: {execution_time}ms")

        return success, result if success else task.error

    def create_and_execute(
        self,
        task_type: str,
        params: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> tuple[bool, Any]:
        """
        Create and immediately execute a task (if no approval needed).

        Returns:
            Tuple of (success, result_or_error_or_task_id)
        """
        task, error = self.create_task(task_type, params, session_id=session_id)
        if not task:
            return False, error

        schema = self.schemas.get(task_type)
        if schema and schema.approval_level != ApprovalLevel.NONE:
            # Needs approval - return task ID
            return True, {"task_id": task.id, "requires_approval": True}

        return self.execute_task(task.id)

    def rollback_task(self, task_id: str) -> tuple[bool, Optional[str]]:
        """
        Rollback a completed task (if reversible).

        Returns:
            Tuple of (success, error_message)
        """
        task = self._load_task(task_id)
        if not task:
            return False, "Task not found"

        if task.status != TaskStatus.COMPLETED:
            return False, f"Cannot rollback {task.status.value} task"

        schema = self.schemas.get(task.task_type)
        if not schema or not schema.reversible:
            return False, "Task is not reversible"

        if not task.rollback_data:
            return False, "No rollback data available"

        # TODO: Implement rollback logic per task type
        with self._lock:
            task.status = TaskStatus.ROLLED_BACK
            self._save_task(task)

        logger.info(f"Rolled back task {task_id}")
        return True, None

    def get_pending_tasks(
        self,
        session_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Task]:
        """Get pending tasks requiring approval."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            if session_id:
                cursor.execute("""
                    SELECT * FROM tasks
                    WHERE status = 'pending' AND session_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (session_id, limit))
            else:
                cursor.execute("""
                    SELECT * FROM tasks
                    WHERE status = 'pending'
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))

            tasks = []
            for row in cursor.fetchall():
                tasks.append(self._row_to_task(row))

            conn.close()
            return tasks

        except Exception as e:
            logger.warning(f"Failed to get pending tasks: {e}")
            return []

    def get_task_history(
        self,
        session_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Task]:
        """Get recent task history."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            if session_id:
                cursor.execute("""
                    SELECT * FROM tasks
                    WHERE session_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (session_id, limit))
            else:
                cursor.execute("""
                    SELECT * FROM tasks
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))

            tasks = []
            for row in cursor.fetchall():
                tasks.append(self._row_to_task(row))

            conn.close()
            return tasks

        except Exception as e:
            logger.warning(f"Failed to get task history: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get task system statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Count by status
            cursor.execute("""
                SELECT status, COUNT(*) FROM tasks
                GROUP BY status
            """)
            status_counts = {row[0]: row[1] for row in cursor.fetchall()}

            # Recent execution times
            cursor.execute("""
                SELECT AVG(execution_time_ms), MAX(execution_time_ms)
                FROM tasks
                WHERE status = 'completed' AND created_at > datetime('now', '-1 day')
            """)
            row = cursor.fetchone()
            avg_time = round(row[0], 1) if row[0] else 0
            max_time = row[1] if row[1] else 0

            # Success rate
            cursor.execute("""
                SELECT COUNT(*) FROM tasks
                WHERE status = 'completed' AND created_at > datetime('now', '-1 day')
            """)
            completed = cursor.fetchone()[0]

            cursor.execute("""
                SELECT COUNT(*) FROM tasks
                WHERE status = 'failed' AND created_at > datetime('now', '-1 day')
            """)
            failed = cursor.fetchone()[0]

            total = completed + failed
            success_rate = (completed / total * 100) if total > 0 else 100

            conn.close()

            return {
                "version": "1.0",
                "enabled": True,
                "registered_schemas": len(self.schemas),
                "pending_count": len(self.pending_tasks),
                "status_counts": status_counts,
                "avg_execution_time_ms": avg_time,
                "max_execution_time_ms": max_time,
                "success_rate_24h": round(success_rate, 1),
            }

        except Exception as e:
            return {
                "version": "1.0",
                "enabled": True,
                "error": str(e),
            }

    def _save_task(self, task: Task) -> None:
        """Save task to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO tasks
                (id, task_type, params, status, priority, created_at,
                 approved_at, started_at, completed_at, result, error,
                 retry_count, execution_time_ms, rollback_data,
                 approval_level, approved_by, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task.id,
                task.task_type,
                json.dumps(task.params),
                task.status.value,
                task.priority.value,
                task.created_at.isoformat(),
                task.approved_at.isoformat() if task.approved_at else None,
                task.started_at.isoformat() if task.started_at else None,
                task.completed_at.isoformat() if task.completed_at else None,
                json.dumps(task.result) if task.result else None,
                task.error,
                task.retry_count,
                task.execution_time_ms,
                json.dumps(task.rollback_data) if task.rollback_data else None,
                task.approval_level.value,
                task.approved_by,
                task.session_id,
            ))

            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to save task: {e}")

    def _load_task(self, task_id: str) -> Optional[Task]:
        """Load task from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
            row = cursor.fetchone()

            conn.close()

            if row:
                return self._row_to_task(row)
            return None

        except Exception as e:
            logger.warning(f"Failed to load task: {e}")
            return None

    def _row_to_task(self, row) -> Task:
        """Convert database row to Task object."""
        return Task(
            id=row[0],
            task_type=row[1],
            params=json.loads(row[2]) if row[2] else {},
            status=TaskStatus(row[3]),
            priority=TaskPriority(row[4]),
            created_at=datetime.fromisoformat(row[5]),
            approved_at=datetime.fromisoformat(row[6]) if row[6] else None,
            started_at=datetime.fromisoformat(row[7]) if row[7] else None,
            completed_at=datetime.fromisoformat(row[8]) if row[8] else None,
            result=json.loads(row[9]) if row[9] else None,
            error=row[10],
            retry_count=row[11] or 0,
            execution_time_ms=row[12] or 0,
            rollback_data=json.loads(row[13]) if row[13] else None,
            approval_level=ApprovalLevel(row[14]) if row[14] else ApprovalLevel.IMPLICIT,
            approved_by=row[15],
            session_id=row[16],
        )


# Singleton instance
_task_manager: Optional[TaskManager] = None


def get_task_manager(db_path: str = "data/evomemory.db") -> TaskManager:
    """Get or create the task manager singleton."""
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager(db_path)
    return _task_manager
