"""Metadaten-Heuristik via exiftool.

Schwaches, aber billiges Signal. Drei Ebenen:
  1. Verräterische KI-Tags (PNG-Chunks von ComfyUI/A1111, Generator-Namen)
     -> starkes KI-Indiz.
  2. Echte Kamera-EXIF (Make/Model + Belichtungsdaten) -> echtes Foto.
  3. Sonst: gestripte Metadaten + KI-typische Auflösung -> leichter Verdacht.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .base import SignalResult, na

NAME = "metadata"
DEFAULT_WEIGHT = 0.7

# Werte/Tag-Namen, die direkt auf Generatoren hindeuten (case-insensitive Teilstring).
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
    "adobe firefly",
    "imagen",
    "gemini",
    "flux",
    "ideogram",
    "leonardo.ai",
    "dreamstudio",
    "invokeai",
)
# PNG-tEXt-Chunk-Namen, die typische Generator-Toolchains hinterlassen.
_AI_TAGNAMES = ("parameters", "workflow", "prompt", "sd-metadata", "dream", "comment")

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
                [tool, "-json", "-G", str(image_path)],
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

        # Tag-Namen ohne Gruppenpräfix (z.B. "PNG:Parameters" -> "parameters").
        plain_keys = {k.split(":")[-1].lower() for k in data}
        blob = json.dumps(data).lower()

        # 1) Verräterische KI-Tags / Generator-Namen.
        kw = next((k for k in _AI_KEYWORDS if k in blob), None)
        tag = next((t for t in _AI_TAGNAMES if t in plain_keys), None)
        if kw:
            return SignalResult(
                self.name, 0.92, self.weight, f"Metadaten nennen KI-Generator: '{kw}'"
            )
        if tag:
            return SignalResult(
                self.name,
                0.85,
                self.weight,
                f"KI-typischer Metadaten-Chunk vorhanden: '{tag}'",
            )

        # 2) Echte Kamera-EXIF.
        cam_hits = [t for t in _CAMERA_TAGS if t in {k.split(":")[-1] for k in data}]
        if len(cam_hits) >= 3:
            return SignalResult(
                self.name,
                0.1,
                self.weight,
                f"Kamera-EXIF vorhanden ({', '.join(cam_hits[:4])})",
            )
        if len(cam_hits) >= 1:
            return SignalResult(
                self.name,
                0.35,
                self.weight,
                f"Teilweise Kamera-EXIF ({', '.join(cam_hits)})",
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
        dim_ai = w in _AI_DIMS and h in _AI_DIMS
        if dim_ai:
            return SignalResult(
                self.name,
                0.65,
                self.weight,
                f"Keine EXIF, KI-typische Auflösung {w}x{h}",
            )
        return SignalResult(
            self.name, 0.55, self.weight, "Keine aussagekräftigen Metadaten (gestript)"
        )


def _as_int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
