from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from typing import Any

from ModelEnvironment.job_context import ContextProvider


@dataclass(frozen=True)
class JournalEntry:
    timestamp: str
    job_id: str
    event: str
    message: str = ""
    data: dict[str, Any] | None = None


class ExecutionJournal:
    def __init__(self, provider: ContextProvider):
        self.provider = provider
        self.journal_path = os.path.join(provider.get().reports_path, "journal.jsonl")

    def append_entry(self, event: str, message: str = "", **data: Any) -> JournalEntry:
        """Append one structured audit entry to the current job's journal.

        The journal deliberately contains orchestration events only. Human-readable
        diagnostics, including runtime stdout and stderr, continue to be handled by
        the regular job loggers.
        """
        context = self.provider.get()
        entry = JournalEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            job_id=context.job_id,
            event=event,
            message=message,
            data=data or None,
        )

        with open(self.journal_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry), sort_keys=True) + "\n")

        return entry
