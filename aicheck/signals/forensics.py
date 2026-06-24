"""Frequenz-Forensik (ohne ML).

Schwaches, erklärbares Zusatzsignal. Misst den Anteil hochfrequenter Energie
im 2D-Powerspektrum. Echte Kamerafotos tragen durch Sensorrauschen und feine
Texturen meist breitbandige Hochfrequenz-Energie; viele (vor allem ältere)
Diffusion-/GAN-Bilder sind im Hochfrequenzbereich glatter oder zeigen
Upsampling-Gitter.

WICHTIG: unzuverlässig. JPEG-Kompression, Downscaling und moderne Generatoren
verwischen das Signal. Daher niedriges Gewicht und ein bewusst enger Score-
Bereich um 0.5 (kaum Eigenmeinung, nur ein leichter Stups).
"""

from __future__ import annotations

from pathlib import Path

from .base import SignalResult, na

NAME = "forensics"
DEFAULT_WEIGHT = 0.5

# Anteil des Spektrumradius, ab dem "hochfrequent" gilt.
_HF_CUTOFF = 0.5
# Empirische Schwellen für den HF-Energieanteil (azimuthal gemittelt).
_HF_LOW = 0.018  # darunter: auffällig glatt -> leicht KI-verdächtig
_HF_HIGH = 0.045  # darüber: viel Hochfrequenz -> eher echte Aufnahme


class ForensicsSignal:
    name = NAME
    weight = DEFAULT_WEIGHT

    def __init__(self, weight: float = DEFAULT_WEIGHT):
        self.weight = weight

    def evaluate(self, image_path: Path) -> SignalResult:
        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:
            return na(self.name, self.weight, f"numpy/PIL fehlt: {exc}", error=True)

        try:
            img = Image.open(image_path).convert("L")
        except Exception as exc:  # noqa: Bilddatei kann beliebig kaputt sein
            return na(self.name, self.weight, f"Bild nicht lesbar: {exc}", error=True)

        # Quadratischer Center-Crop + Begrenzung auf 1024 px für Tempo.
        side = min(img.size)
        side = min(side, 1024)
        left = (img.width - side) // 2
        top = (img.height - side) // 2
        img = img.crop((left, top, left + side, top + side))

        arr = np.asarray(img, dtype=np.float64)
        if arr.size == 0 or arr.std() < 1e-6:
            return na(self.name, self.weight, "Bild ohne verwertbare Struktur")

        # 2D-FFT-Powerspektrum, zentriert.
        f = np.fft.fftshift(np.fft.fft2(arr))
        power = np.abs(f) ** 2
        total = power.sum()
        if total <= 0:
            return na(self.name, self.weight, "kein Spektrum berechenbar")

        # Radialdistanz jedes Pixels vom Zentrum, normiert auf [0,1].
        h, w = power.shape
        cy, cx = h / 2.0, w / 2.0
        yy, xx = np.ogrid[:h, :w]
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        r /= r.max()

        hf_fraction = float(power[r >= _HF_CUTOFF].sum() / total)

        # Auf engen Score-Bereich abbilden (0.40 .. 0.60).
        if hf_fraction <= _HF_LOW:
            score, verdict = 0.60, "auffällig wenig Hochfrequenz-Energie"
        elif hf_fraction >= _HF_HIGH:
            score, verdict = 0.40, "viel Hochfrequenz-Energie (eher Aufnahme)"
        else:
            # linear zwischen den Schwellen interpolieren
            t = (hf_fraction - _HF_LOW) / (_HF_HIGH - _HF_LOW)
            score = 0.60 - 0.20 * t
            verdict = "Hochfrequenzanteil im neutralen Bereich"

        return SignalResult(
            self.name,
            round(score, 3),
            self.weight,
            f"{verdict} (HF-Anteil={hf_fraction:.4f})",
        )
