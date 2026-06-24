"""C2PA / Content-Credentials-Signal.

Liest mit `c2patool` ein eingebettetes C2PA-Manifest. Das ist das einzige
*harte* Signal: viele Generatoren (OpenAI/GPT-Image, Adobe Firefly, Google
Gemini/Imagen) und Kameras betten einen IPTC `digitalSourceType` ein.

  trainedAlgorithmicMedia / compositeWithTrainedAlgorithmicMedia -> KI
  digitalCapture / digitalCreation(Foto)                          -> echte Aufnahme

Fehlt das Manifest (Normalfall bei gestripten Bildern) -> score=None.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .base import SignalResult, na

NAME = "c2pa"
DEFAULT_WEIGHT = 3.0  # hohes Gewicht, greift aber nur bei vorhandenem Manifest

# IPTC digitalSourceType-Schlüsselwörter (Teilstrings, case-insensitive)
_AI_SOURCE_TYPES = (
    "trainedalgorithmicmedia",
    "compositewithtrainedalgorithmicmedia",
    "algorithmicmedia",
)
_REAL_SOURCE_TYPES = ("digitalcapture",)
# Bekannte KI-Generatoren im claim_generator-String
_AI_GENERATORS = (
    "openai",
    "dall",
    "gpt-image",
    "firefly",
    "midjourney",
    "imagen",
    "gemini",
    "stable diffusion",
    "stability",
    "flux",
    "ideogram",
    "leonardo",
    "runway",
    "adobe firefly",
)


class C2paSignal:
    name = NAME
    weight = DEFAULT_WEIGHT

    def __init__(self, weight: float = DEFAULT_WEIGHT):
        self.weight = weight

    def evaluate(self, image_path: Path) -> SignalResult:
        tool = shutil.which("c2patool")
        if not tool:
            return na(
                self.name,
                self.weight,
                "c2patool nicht installiert (install c2patool)",
                error=True,
            )
        try:
            proc = subprocess.run(
                [tool, str(image_path)],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            return na(self.name, self.weight, f"c2patool-Fehler: {exc}", error=True)

        out = (proc.stdout or "").strip()
        if proc.returncode != 0 or not out:
            # Häufigster Fall: "No claim found" -> kein Manifest, kein Urteil.
            err = (proc.stderr or "").strip()
            if "no claim" in err.lower() or "no claim" in out.lower():
                return na(self.name, self.weight, "kein C2PA-Manifest eingebettet")
            return na(
                self.name, self.weight, f"kein verwertbares Manifest ({err or 'leer'})"
            )

        try:
            data = json.loads(out)
        except json.JSONDecodeError:
            return na(
                self.name, self.weight, "Manifest nicht als JSON lesbar", error=True
            )

        blob = json.dumps(data).lower()
        source_types = _collect(data, "digitalsourcetype") + _collect(
            data, "digital_source_type"
        )
        st_blob = " ".join(source_types).lower()

        ai_hit = any(t in st_blob for t in _AI_SOURCE_TYPES)
        real_hit = any(t in st_blob for t in _REAL_SOURCE_TYPES)
        gen_hit = next((g for g in _AI_GENERATORS if g in blob), None)

        if ai_hit:
            return SignalResult(
                self.name,
                1.0,
                self.weight,
                f"C2PA-Manifest weist KI-Quelle aus (digitalSourceType: {st_blob})",
                hard_verdict="ai",
            )
        if real_hit and not gen_hit:
            return SignalResult(
                self.name,
                0.05,
                self.weight,
                "C2PA-Manifest weist echte Aufnahme aus (digitalCapture)",
                hard_verdict="real",
            )
        if gen_hit:
            return SignalResult(
                self.name,
                0.9,
                self.weight,
                f"C2PA-Manifest nennt KI-Generator '{gen_hit}'",
                hard_verdict="ai",
            )
        # Manifest vorhanden, aber kein eindeutiger Herkunfts-Hinweis.
        return SignalResult(
            self.name, 0.5, self.weight, "C2PA-Manifest vorhanden, Herkunft unklar"
        )


def _collect(obj, key: str) -> list[str]:
    """Sammelt rekursiv alle String-Werte unter einem (case-insensitiven) Key."""
    found: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() == key and isinstance(v, str):
                found.append(v)
            else:
                found.extend(_collect(v, key))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(_collect(item, key))
    return found
