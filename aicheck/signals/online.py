"""Online-Detektor (optional, nur mit --online).

Standard-Provider: **Sightengine** (https://sightengine.com). Aktivierung:

    export SIGHTENGINE_API_USER=...
    export SIGHTENGINE_API_SECRET=...
    aicheck bild.png --online

Ohne --online verlässt kein Bild den Rechner. Weitere Provider lassen sich in
_call_provider() ergänzen (Adapter-Muster). Die Antwort-Auswertung steckt in
der reinen, testbaren Funktion _parse_sightengine().
"""

from __future__ import annotations

import os
from pathlib import Path

from .. import cache as cache_mod
from .base import SignalResult, na

NAME = "online"
DEFAULT_WEIGHT = 2.0
SIGHTENGINE_ENDPOINT = "https://api.sightengine.com/1.0/check.json"


class _ConfigError(Exception):
    """Konfigurationsproblem (fehlende Credentials etc.) -> als n/a melden."""


class OnlineSignal:
    name = NAME
    weight = DEFAULT_WEIGHT

    def __init__(
        self,
        weight: float = DEFAULT_WEIGHT,
        enabled: bool = False,
        provider: str | None = None,
        refresh: bool = False,
    ):
        self.weight = weight
        self.enabled = enabled
        self.provider = provider or os.environ.get(
            "AICHECK_ONLINE_PROVIDER", "sightengine"
        )
        self.refresh = refresh
        self._cache: cache_mod.OnlineCache | None = None

    def _cache_obj(self) -> cache_mod.OnlineCache:
        if self._cache is None:
            self._cache = cache_mod.OnlineCache()
        return self._cache

    def evaluate(self, image_path: Path) -> SignalResult:
        if not self.enabled:
            return na(
                self.name,
                self.weight,
                "Online-Check deaktiviert (--online zum Aktivieren)",
            )

        # 1) Cache-Lookup über den Bild-Hash (spart einen kostenpflichtigen Call).
        #    Treffer brauchen keine API-Keys.
        try:
            file_hash = cache_mod.image_hash(image_path)
        except OSError as exc:
            return na(self.name, self.weight, f"Bild nicht lesbar: {exc}", error=True)
        cache = self._cache_obj()
        if not self.refresh:
            hit = cache.get(self.provider, file_hash)
            if hit is not None and hit.get("score") is not None:
                return SignalResult(
                    self.name,
                    float(hit["score"]),
                    self.weight,
                    f"{hit.get('detail', '')} (aus Cache, 0 Ops)",
                )

        # 2) Kein Treffer (oder --refresh) -> echter, kostenpflichtiger Call.
        try:
            score, detail = self._call_provider(self.provider, image_path)
        except _ConfigError as exc:
            return na(self.name, self.weight, str(exc), error=True)
        except Exception as exc:  # noqa: Netzfehler aller Art kapseln
            return na(
                self.name,
                self.weight,
                f"Online-Fehler ({self.provider}): {exc}",
                error=True,
            )
        if score is None:
            return na(self.name, self.weight, detail, error=True)  # Fehler NICHT cachen

        cache.put(self.provider, file_hash, {"score": float(score), "detail": detail})
        return SignalResult(self.name, float(score), self.weight, detail)

    def _call_provider(
        self, provider: str, image_path: Path
    ) -> tuple[float | None, str]:
        if provider == "sightengine":
            return _sightengine(image_path)
        raise _ConfigError(f"unbekannter Online-Provider '{provider}'")


def _sightengine(image_path: Path) -> tuple[float | None, str]:
    user = os.environ.get("SIGHTENGINE_API_USER")
    secret = os.environ.get("SIGHTENGINE_API_SECRET")
    if not user or not secret:
        raise _ConfigError(
            "SIGHTENGINE_API_USER/SIGHTENGINE_API_SECRET nicht gesetzt "
            "(siehe .env.example)"
        )
    try:
        import requests
    except ImportError:
        raise _ConfigError("Paket 'requests' fehlt -> pip install requests")

    with open(image_path, "rb") as fh:
        resp = requests.post(
            SIGHTENGINE_ENDPOINT,
            files={"media": fh},
            data={"models": "genai", "api_user": user, "api_secret": secret},
            timeout=30,
        )
    return _parse_sightengine(resp.json())


def _parse_sightengine(data: dict) -> tuple[float | None, str]:
    """Wertet eine Sightengine-Antwort aus -> (ai_score 0..1 | None, detail).
    Rein und ohne I/O, daher direkt testbar."""
    if data.get("status") != "success":
        err = data.get("error")
        msg = (
            err.get("message")
            if isinstance(err, dict)
            else (err or "unbekannter Fehler")
        )
        return None, f"Sightengine-Fehler: {msg}"

    type_block = data.get("type") or {}
    score = type_block.get("ai_generated")
    if score is None:
        return None, "Sightengine: Feld type.ai_generated fehlt"

    top = ""
    generators = type_block.get("ai_generators")
    if isinstance(generators, dict) and generators:
        name, val = max(
            generators.items(),
            key=lambda kv: kv[1] if isinstance(kv[1], (int, float)) else 0.0,
        )
        if isinstance(val, (int, float)) and val > 0:
            top = f", top: {name} {val:.2f}"

    return float(score), f"Sightengine: ai_generated={float(score):.2f}{top}"
