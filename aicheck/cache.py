"""Lokaler Ergebnis-Cache für kostenpflichtige Online-Calls.

Schlüssel = provider + SHA-256 des Bild*inhalts*. Damit greift der Cache auch
nach Umbenennen, und ein verändertes Bild bekommt automatisch einen neuen Call.
Nur erfolgreiche Ergebnisse werden gespeichert (Fehler/None nie), damit ein
transienter API-Fehler nicht dauerhaft hängen bleibt.

Speicherort: ~/.aicheck/online-cache.json (ausserhalb des Repos).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CACHE_DIR = Path.home() / ".aicheck"
CACHE_FILE = CACHE_DIR / "online-cache.json"


def image_hash(image_path) -> str:
    """SHA-256 über den Dateiinhalt (streamend, auch für grosse Bilder)."""
    h = hashlib.sha256()
    with open(image_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


class OnlineCache:
    """Lädt den Cache einmal in den Speicher; schreibt nach jedem put sofort
    durch (write-through), damit teuer bezahlte Ergebnisse keinen Crash überleben
    müssen."""

    def __init__(self, path: Path = CACHE_FILE):
        self.path = path
        self._data = _load(path)

    @staticmethod
    def key(provider: str, file_hash: str) -> str:
        return f"{provider}:{file_hash}"

    def get(self, provider: str, file_hash: str) -> dict | None:
        return self._data.get(self.key(provider, file_hash))

    def put(self, provider: str, file_hash: str, entry: dict) -> None:
        self._data[self.key(provider, file_hash)] = entry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False))


def clear(path: Path = CACHE_FILE) -> int:
    """Leert den Cache. Gibt die Anzahl entfernter Einträge zurück."""
    n = len(_load(path))
    if path.exists():
        path.unlink()
    return n
