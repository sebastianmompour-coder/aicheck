"""Lokales ML-Signal — Hauptsignal.

Nutzt einen HuggingFace-Bildklassifikator (image-classification pipeline), der
zwischen KI-generierten und echten Bildern unterscheidet. Das Modell ist über
--model bzw. den Konstruktor frei tauschbar (Detektoren veralten schnell).

Robuste Label-Zuordnung: das Modell kann seine Klassen beliebig benennen
("fake"/"real", "artificial"/"human", "ai"/"hum" ...). Wir erkennen die
KI-Klasse anhand des Labelnamens und nehmen deren Wahrscheinlichkeit als Score.
"""

from __future__ import annotations

from pathlib import Path

from .base import SignalResult, na

NAME = "ml_local"
DEFAULT_WEIGHT = 2.0

# Default-Modell. Empirisch gegen echte Referenzfotos vs. Verdachtsbilder
# gewählt: beste Trennung bei niedrigster Falsch-Positiv-Rate.
#   haywoodsloan/...: echt mean P(KI)=0.13 / KI mean=0.99, FP 1/8
#   Organika/sdxl-detector (Alternative): echt 0.24 / KI 1.00, FP 2/8
DEFAULT_MODEL = "haywoodsloan/ai-image-detector-deploy"

# Label-Schlüsselwörter -> Klasse.
_AI_LABELS = (
    "ai",
    "artificial",
    "fake",
    "gan",
    "generated",
    "synthetic",
    "deepfake",
    "diffusion",
    "computer",
    "machine",
)
_REAL_LABELS = (
    "real",
    "human",
    "authentic",
    "genuine",
    "natural",
    "camera",
    "photo",
    "nature",
    "hum",
)


class MlLocalSignal:
    name = NAME
    weight = DEFAULT_WEIGHT

    def __init__(self, weight: float = DEFAULT_WEIGHT, model: str = DEFAULT_MODEL):
        self.weight = weight
        self.model = model
        self._pipe = None  # lazy

    def _ensure_pipe(self):
        if self._pipe is not None:
            return self._pipe
        import logging
        import os
        import sys

        # HF-Hub-Logwarnungen (z.B. "unauthenticated requests") sind hier
        # irrelevant: geladen wird nur das Modell, niemals Bilddaten.
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

        # Liegt das Modell schon lokal im Cache, dann komplett offline arbeiten:
        # kein Netzwerk, kein Versions-Check, keine Warnung. Sonst einmalig laden.
        if _is_cached(self.model):
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            try:  # auch direkt patchen, falls huggingface_hub schon importiert ist
                from huggingface_hub import constants as _hfc

                _hfc.HF_HUB_OFFLINE = True
            except Exception:
                pass
        else:
            print(
                f"[aicheck] Lade Detektormodell '{self.model}' einmalig vom "
                "HuggingFace-Hub herunter (danach läuft alles offline) ...",
                file=sys.stderr,
            )

        from transformers.utils import logging as tlog

        tlog.set_verbosity_error()
        from transformers import pipeline

        self._pipe = pipeline("image-classification", model=self.model, device="cpu")
        return self._pipe

    def evaluate(self, image_path: Path) -> SignalResult:
        try:
            from PIL import Image

            pipe = self._ensure_pipe()
        except Exception as exc:  # noqa: Modell-Load/Download kann vielfältig scheitern
            return na(
                self.name,
                self.weight,
                f"Modell nicht ladbar ({self.model}): {exc}",
                error=True,
            )

        try:
            img = Image.open(image_path).convert("RGB")
            preds = pipe(img, top_k=None)
        except Exception as exc:
            return na(
                self.name, self.weight, f"Inferenz fehlgeschlagen: {exc}", error=True
            )

        score = _ai_probability(preds)
        if score is None:
            labels = ", ".join(p.get("label", "?") for p in preds)
            return na(
                self.name,
                self.weight,
                f"Labels nicht eindeutig zuordenbar ({labels})",
                error=True,
            )

        top = max(preds, key=lambda p: p.get("score", 0.0))
        return SignalResult(
            self.name,
            round(float(score), 4),
            self.weight,
            f"P(KI)={score:.2f} (Modell-Label '{top.get('label')}'"
            f"={top.get('score', 0):.2f}) [{self.model}]",
        )


def _is_cached(model_id: str) -> bool:
    """True, wenn das Modell bereits im lokalen HF-Cache liegt (kein Download
    nötig). Bewusst tolerant: im Zweifel False -> normaler Lade-/Downloadpfad."""
    try:
        from huggingface_hub import try_to_load_from_cache

        path = try_to_load_from_cache(repo_id=model_id, filename="config.json")
        return isinstance(path, str)
    except Exception:
        return False


def _ai_probability(preds: list[dict]) -> float | None:
    """Summiert die Wahrscheinlichkeit aller als 'KI' erkannten Labels."""
    ai_p = 0.0
    real_p = 0.0
    matched = False
    for p in preds:
        label = str(p.get("label", "")).lower()
        prob = float(p.get("score", 0.0))
        if any(k in label for k in _AI_LABELS):
            ai_p += prob
            matched = True
        elif any(k in label for k in _REAL_LABELS):
            real_p += prob
            matched = True
    if not matched:
        return None
    denom = ai_p + real_p
    if denom <= 0:
        return None
    return ai_p / denom
