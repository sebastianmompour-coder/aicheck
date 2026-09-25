"""Metadaten-Heuristik via exiftool.

Schwaches, aber billiges Signal. Drei Ebenen:
  1. Verräterische KI-Tags (PNG-Chunks von ComfyUI/A1111, Generator-Namen,
     SD-Parameterstrings) -> starkes KI-Indiz.
  2. Echte Kamera-EXIF (Make/Model + Belichtungsdaten) -> echtes Foto.
  3. Sonst: gestripte Metadaten + KI-typische Auflösung -> leichter Verdacht.

Der Subprocess-Aufruf steckt in evaluate(); die Bewertung selbst ist die reine
Funktion assess_exif() und damit ohne exiftool testbar.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from .base import SignalResult, find_keyword, na

NAME = "metadata"
DEFAULT_WEIGHT = 0.7

# Generator-Namen (Wortgrenzen-Match in Metadaten-Werten, case-insensitiv).
_AI_KEYWORDS = (
    "midjourney",
    "dall-e",
    "dall·e",
    "dalle",
    "openai",
    "gpt-image",
    "stable diffusion",
    "stable-diffusion",
    "automatic1111",
    "comfyui",
    "novelai",
    "firefly",
    "imagen",
    "gemini",
    "flux",
    "ideogram",
    "leonardo.ai",
    "dreamstudio",
    "invokeai",
)
# PNG-tEXt-Chunk-Namen, die typische Generator-Toolchains hinterlassen.
# Bewusst NICHT "comment": das JPEG-COM-Segment (exiftool "File:Comment")
# schreiben auch PHP/GD, ImageMagick & Co. — das wäre ein Dauer-False-Positive.
_AI_TAGNAMES = ("parameters", "workflow", "prompt", "sd-metadata", "dream")
# A1111/NovelAI-Parameterstring ("Steps: 20, Sampler: Euler a, CFG scale: 7 ..."),
# egal in welchem Feld er landet (PNG:Parameters, EXIF:UserComment, Comment).
_SD_PARAMS_RE = re.compile(r"\bsteps:\s*\d+.*?\bsampler:", re.IGNORECASE | re.DOTALL)

# Felder, die den Dateisystem-Pfad enthalten. Sie dürfen NIE in die
# Keyword-Suche einfliessen — sonst macht ein Ordner "gemini/" jedes Bild
# darin KI-verdächtig.
_PATH_KEYS = {"sourcefile", "filename", "directory"}

# Kamera-Belegtags: ihr Vorhandensein spricht stark für eine echte Aufnahme.
_CAMERA_TAGS = (
    "Make",
    "Model",
    "ExposureTime",
    "FNumber",
    "ISO",
    "ISOSpeed",
    "DateTimeOriginal",
    "LensModel",
    "FocalLength",
    "GPSLatitude",
)

# Häufige KI-Ausgabe-Auflösungen (Breite/Höhe-Werte).
_AI_DIMS = {
    512,
    576,
    640,
    704,
    768,
    832,
    896,
    1024,
    1152,
    1216,
    1280,
    1344,
    1408,
    1536,
    1664,
    1792,
    1856,
    2048,
}


class MetadataSignal:
    name = NAME
    weight = DEFAULT_WEIGHT

    def __init__(self, weight: float = DEFAULT_WEIGHT):
        self.weight = weight

    def evaluate(self, image_path: Path) -> SignalResult:
        tool = shutil.which("exiftool")
        if not tool:
            return na(self.name, self.weight, "exiftool nicht installiert", error=True)
        try:
            proc = subprocess.run(
                # Absoluter Pfad: ein Dateiname wie "-x.png" darf nie als
                # exiftool-Option interpretiert werden.
                [tool, "-json", "-G", str(image_path.resolve())],
                capture_output=True,
                text=True,
                timeout=30,
            )
            data = json.loads(proc.stdout)[0]
        except (
            subprocess.TimeoutExpired,
            OSError,
            json.JSONDecodeError,
            IndexError,
        ) as exc:
            return na(self.name, self.weight, f"exiftool-Fehler: {exc}", error=True)

        return assess_exif(data, name=self.name, weight=self.weight)


def assess_exif(
    data: dict, name: str = NAME, weight: float = DEFAULT_WEIGHT
) -> SignalResult:
    """Bewertet eine exiftool-JSON-Ausgabe (`exiftool -json -G`, ein Objekt).
    Rein und ohne I/O, daher direkt testbar."""
    # Tag-Namen ohne Gruppenpräfix (z.B. "PNG:Parameters" -> "parameters").
    plain_keys = {_plain(k) for k in data}
    # Nur Metadaten-WERTE durchsuchen, ohne Pfad-Felder und ohne Tag-Namen.
    text = " ".join(
        _flatten(v) for k, v in data.items() if _plain(k).lower() not in _PATH_KEYS
    )

    # 1) Verräterische KI-Tags / Generator-Namen / SD-Parameterstring.
    kw = find_keyword(text, _AI_KEYWORDS)
    if kw:
        return SignalResult(name, 0.92, weight, f"Metadaten nennen KI-Generator: '{kw}'")
    tag = next((t for t in _AI_TAGNAMES if t in plain_keys), None)
    if tag:
        return SignalResult(
            name, 0.85, weight, f"KI-typischer Metadaten-Chunk vorhanden: '{tag}'"
        )
    if _SD_PARAMS_RE.search(text):
        return SignalResult(
            name, 0.85, weight, "Stable-Diffusion-Parameterstring in Metadaten"
        )

    # 2) Echte Kamera-EXIF.
    present = {k.split(":")[-1] for k in data}
    cam_hits = [t for t in _CAMERA_TAGS if t in present]
    if len(cam_hits) >= 3:
        return SignalResult(
            name, 0.1, weight, f"Kamera-EXIF vorhanden ({', '.join(cam_hits[:4])})"
        )
    if len(cam_hits) >= 1:
        return SignalResult(
            name, 0.35, weight, f"Teilweise Kamera-EXIF ({', '.join(cam_hits)})"
        )

    # 3) Keine aussagekräftigen Metadaten -> Auflösung als schwacher Hinweis.
    w = _as_int(
        data.get("EXIF:ImageWidth")
        or data.get("File:ImageWidth")
        or data.get("PNG:ImageWidth")
    )
    h = _as_int(
        data.get("EXIF:ImageHeight")
        or data.get("File:ImageHeight")
        or data.get("PNG:ImageHeight")
    )
    if w in _AI_DIMS and h in _AI_DIMS:
        return SignalResult(
            name, 0.65, weight, f"Keine EXIF, KI-typische Auflösung {w}x{h}"
        )
    return SignalResult(
        name, 0.55, weight, "Keine aussagekräftigen Metadaten (gestript)"
    )


def _plain(key: str) -> str:
    return key.split(":")[-1].lower()


def _flatten(v) -> str:
    """Wert (auch verschachtelt) als durchsuchbaren Text."""
    if isinstance(v, str):
        return v
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _as_int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
