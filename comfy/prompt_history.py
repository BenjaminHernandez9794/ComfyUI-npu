from __future__ import annotations

import json
import logging
import os
import sqlite3


class PromptHistoryStore:
    """Durable storage for completed prompt history."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prompt_history (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_id TEXT NOT NULL UNIQUE,
                payload TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def load(self) -> dict:
        history = {}
        rows = self._connection.execute(
            "SELECT prompt_id, payload FROM prompt_history ORDER BY sequence"
        )
        for prompt_id, payload in rows:
            try:
                history[prompt_id] = json.loads(payload)
            except (TypeError, ValueError):
                logging.exception("Skipping invalid persisted history item %s", prompt_id)
        return history

    def save(self, prompt_id: str, history_item: dict) -> None:
        payload = json.dumps(history_item, ensure_ascii=False, separators=(",", ":"))
        if sqlite3.sqlite_version_info >= (3, 24, 0):
            self._connection.execute(
                """
                INSERT INTO prompt_history (prompt_id, payload)
                VALUES (?, ?)
                ON CONFLICT(prompt_id) DO UPDATE SET payload = excluded.payload
                """,
                (prompt_id, payload),
            )
        else:
            # SQLite < 3.24 (e.g. some CANN/buildtools Python builds) has no UPSERT syntax
            cur = self._connection.execute(
                "UPDATE prompt_history SET payload = ? WHERE prompt_id = ?", (payload, prompt_id)
            )
            if cur.rowcount == 0:
                self._connection.execute(
                    "INSERT INTO prompt_history (prompt_id, payload) VALUES (?, ?)", (prompt_id, payload)
                )
        self._connection.commit()

    def delete(self, prompt_id: str) -> None:
        self._connection.execute(
            "DELETE FROM prompt_history WHERE prompt_id = ?", (prompt_id,)
        )
        self._connection.commit()

    def clear(self) -> None:
        self._connection.execute("DELETE FROM prompt_history")
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()
