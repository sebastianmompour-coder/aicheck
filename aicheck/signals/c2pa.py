"""C2PA / Content-Credentials-Signal.

Liest mit `c2patool` ein eingebettetes C2PA-Manifest. Das ist das einzige
*harte* Signal: viele Generatoren (OpenAI/GPT-Image, Adobe Firefly, Google
Gemini/Imagen) und Kameras betten einen IPTC `digitalSourceType` ein.

  trainedAlgorithmicMedia / compositeWithTrainedAlgorithmicMedia -> KI
  digitalCapture / digitalCreation(Foto)                          -> echte Aufnahme

Fehlt das Manifest (Normalfall bei gestripten Bildern) -> score=None.

Der Subprocess-Aufruf steckt in evaluate(); die Bewertung selbst ist die reine
Funktion assess_manifest() und damit ohne c2patool testbar. Generator-Namen
werden nur in den Herkunftsfeldern gesucht (claim_generator, softwareAgent),
nie im Asset-Titel oder in Zertifikatsnamen.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .base import SignalResult, find_keyword, na

NAME = "c2pa"
DEFAULT_WEIGHT = 3.0  # hohes Gewicht, greift aber nur bei vorhandenem Manifest

# IPTC digitalSourceType-Schlüsselwörter (Teilstrings der URI, case-insensitive)
_AI_SOURCE_TYPES = (
    "trainedalgorithmicmedia",
    "compositewithtrainedalgorithmicmedia",
    "algorithmicmedia",
)
_REAL_SOURCE_TYPES = ("digitalcapture",)
# Bekannte KI-Generatoren (Wortgrenzen-Match in den Generator-Feldern)
_AI_GENERATORS = (
    "openai",
    "dall-e",
    "dall·e",
    "dalle",
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
)
# Felder, in denen der erzeugende Software-Agent steht.
_GENERATOR_KEYS = ("claim_generator", "claim_generator_info", "softwareagent")


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
                # Absoluter Pfad: ein Dateiname wie "-x.png" darf nie als
                # c2patool-Option interpretiert werden.
                [tool, str(image_path.resolve())],
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

        return assess_manifest(data, name=self.name, weight=self.weight)


def assess_manifest(
    data: dict, name: str = NAME, weight: float = DEFAULT_WEIGHT
) -> SignalResult:
    """Bewertet eine c2patool-JSON-Ausgabe. Rein und ohne I/O, daher testbar."""
    source_types = _collect(data, "digitalsourcetype") + _collect(
        data, "digital_source_type"
    )
    st_blob = " ".join(source_types).lower()
    generators = " ".join(_generator_strings(data))

    ai_hit = any(t in st_blob for t in _AI_SOURCE_TYPES)
    real_hit = any(t in st_blob for t in _REAL_SOURCE_TYPES)
    gen_hit = find_keyword(generators, _AI_GENERATORS)

    if ai_hit:
        return SignalResult(
            name,
            1.0,
            weight,
            f"C2PA-Manifest weist KI-Quelle aus (digitalSourceType: {st_blob})",
            hard_verdict="ai",
        )
    if real_hit and not gen_hit:
        return SignalResult(
            name,
            0.05,
            weight,
            "C2PA-Manifest weist echte Aufnahme aus (digitalCapture)",
            hard_verdict="real",
        )
    if gen_hit:
        return SignalResult(
            name,
            0.9,
            weight,
            f"C2PA-Manifest nennt KI-Generator '{gen_hit}'",
            hard_verdict="ai",
        )
    # Manifest vorhanden, aber kein eindeutiger Herkunfts-Hinweis.
    return SignalResult(name, 0.5, weight, "C2PA-Manifest vorhanden, Herkunft unklar")


def _generator_strings(data) -> list[str]:
    """Alle Strings aus den Generator-Feldern (claim_generator,
    claim_generator_info[].name, actions[].softwareAgent(.name))."""
    out: list[str] = []
    for key in _GENERATOR_KEYS:
        for value in _collect_values(data, key):
            out.extend(_strings_in(value))
    return out


def _strings_in(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        name = value.get("name")
        return [name] if isinstance(name, str) else []
    if isinstance(value, list):
        return [s for item in value for s in _strings_in(item)]
    return []


def _collect_values(obj, key: str) -> list:
    """Sammelt rekursiv alle Werte unter einem (case-insensitiven) Key. Ein
    XMP-Namespace-Präfix wird ignoriert: "Iptc4xmpExt:DigitalSourceType"
    zählt als "digitalsourcetype"."""
    found: list = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower().rsplit(":", 1)[-1] == key:
                found.append(v)
            else:
                found.extend(_collect_values(v, key))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(_collect_values(item, key))
    return found


def _collect(obj, key: str) -> list[str]:
    """Sammelt rekursiv alle String-Werte unter einem (case-insensitiven) Key."""
    return [v for v in _collect_values(obj, key) if isinstance(v, str)]
