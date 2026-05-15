"""In-memory progress tracker.

Future iterations may expose this over HTTP. For now it's a thread-safe-ish
process-local dict used by news_viral_pro.py tasks to publish their state.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional


@dataclass
class TaskProgress:
    name: str
    status: str = "pending"  # pending | running | done | failed
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    message: str = ""


@dataclass
class ProgressTracker:
    """Records progress for each pipeline task. Process-local, no persistence."""
    tasks: Dict[str, TaskProgress] = field(default_factory=dict)

    def start(self, name: str, message: str = "") -> None:
        self.tasks[name] = TaskProgress(
            name=name,
            status="running",
            started_at=datetime.now().isoformat(),
            message=message,
        )

    def done(self, name: str, message: str = "") -> None:
        t = self.tasks.get(name) or TaskProgress(name=name)
        t.status = "done"
        t.ended_at = datetime.now().isoformat()
        t.message = message
        self.tasks[name] = t

    def fail(self, name: str, message: str) -> None:
        t = self.tasks.get(name) or TaskProgress(name=name)
        t.status = "failed"
        t.ended_at = datetime.now().isoformat()
        t.message = message
        self.tasks[name] = t

    def snapshot(self) -> Dict[str, Dict]:
        return {k: v.__dict__ for k, v in self.tasks.items()}
