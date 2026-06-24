"""Gemeinsame Schnittstelle aller Erkennungs-Signale.

Jedes Signal bewertet ein Bild unabhängig und liefert ein SignalResult.
Score-Konvention: 0.0 = sicher echt ... 1.0 = sicher KI-generiert.
score=None bedeutet "Signal nicht anwendbar / kein Urteil" und fällt aus
der gewichteten Gesamtbewertung heraus.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class SignalResult:
    """Ergebnis eines einzelnen Signals für ein Bild."""

    name: str
    score: float | None  # 0.0 echt ... 1.0 KI; None = nicht anwendbar
    weight: float  # Gewicht im Gesamtscore (0 = ignorieren)
    detail: str  # menschenlesbare Begründung
    error: bool = False  # True, wenn das Signal technisch fehlschlug
    # Optionales hartes Override: "ai" oder "real" erzwingt die Ampel,
    # unabhängig vom gewichteten Score (z.B. valides C2PA-Manifest).
    hard_verdict: str | None = None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "score": self.score,
            "weight": self.weight,
            "detail": self.detail,
            "error": self.error,
            "hard_verdict": self.hard_verdict,
        }


@runtime_checkable
class Signal(Protocol):
    """Protokoll für ein Signal. name/weight als Attribute, evaluate() liefert
    das Ergebnis. Implementierungen kapseln ihre eigenen Fehler und geben im
    Fehlerfall score=None + error=True zurück, statt zu werfen."""

    name: str
    weight: float

    def evaluate(self, image_path: Path) -> SignalResult: ...


def na(name: str, weight: float, detail: str, error: bool = False) -> SignalResult:
    """Hilfsfunktion: 'nicht anwendbar'-Ergebnis."""
    return SignalResult(
        name=name, score=None, weight=weight, detail=detail, error=error
    )
