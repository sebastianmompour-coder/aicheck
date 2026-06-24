"""Aggregation der Signale zu einer Ampel.

Gesamtscore = gewichteter Mittelwert aller anwendbaren Signal-Scores
(score=None fällt heraus). Ein Signal mit hard_verdict erzwingt die Ampel
unabh#ngig vom Score (z.B. valides C2PA-Manifest).

Ampel: GREEN (echt-wahrscheinlich) / AMBER (unklar) / RED (KI-verdächtig).
Triage-orientiert: im Zweifel eher AMBER/RED.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .signals.base import SignalResult

GREEN, AMBER, RED = "GREEN", "AMBER", "RED"
_EMOJI = {GREEN: "🟢", AMBER: "🟡", RED: "🔴"}
_LABEL = {GREEN: "echt-wahrscheinlich", AMBER: "unklar", RED: "KI-verd#chtig"}


@dataclass
class Verdict:
    level: str  # GREEN / AMBER / RED
    score: float | None  # gewichteter Gesamtscore (None wenn keine Signale)
    reason: str  # Hauptbegründung
    signals: list[SignalResult] = field(default_factory=list)

    @property
    def emoji(self) -> str:
        return _EMOJI[self.level]

    @property
    def label(self) -> str:
        return _LABEL[self.level]


def aggregate(
    signals: list[SignalResult],
    amber_threshold: float = 0.35,
    red_threshold: float = 0.60,
) -> Verdict:
    # 1) Harte Overrides (C2PA) haben Vorrang.
    ai_hard = [s for s in signals if s.hard_verdict == "ai"]
    real_hard = [s for s in signals if s.hard_verdict == "real"]
    if ai_hard:
        return Verdict(RED, 1.0, ai_hard[0].detail, signals)
    if real_hard:
        return Verdict(GREEN, 0.0, real_hard[0].detail, signals)

    # 2) Gewichteter Mittelwert über anwendbare Signale.
    usable = [s for s in signals if s.score is not None and s.weight > 0]
    if not usable:
        return Verdict(
            AMBER, None, "kein Signal lieferte ein Urteil — manuell prüfen", signals
        )

    total_w = sum(s.weight for s in usable)
    score = sum(s.score * s.weight for s in usable) / total_w

    if score >= red_threshold:
        level = RED
    elif score >= amber_threshold:
        level = AMBER
    else:
        level = GREEN

    reason = _dominant_reason(usable, level)
    return Verdict(level, round(score, 4), reason, signals)


def _dominant_reason(usable: list[SignalResult], level: str) -> str:
    """Begründung aus dem Signal mit dem grössten gewichteten Beitrag in
    Richtung des Urteils."""
    if level == GREEN:
        key = lambda s: (1.0 - s.score) * s.weight  # noqa: E731
    else:
        key = lambda s: s.score * s.weight  # noqa: E731
    top = max(usable, key=key)
    return f"{top.name}: {top.detail}"
