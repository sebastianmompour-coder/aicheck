"""Tests für aicheck. Läuft ohne pytest:  .venv/bin/python -m unittest -v

Kernlogik (Scoring, Label-Mapping) wird synthetisch getestet (schnell, kein I/O).
Die optionalen Signal-Smoketests laufen gegen eigene Bildordner und werden
übersprungen, falls die Ordner fehlen. Pfade über Umgebungsvariablen setzen:

  AICHECK_TEST_AI_DIR   = Ordner mit (vermutlich) KI-Bildern
  AICHECK_TEST_REAL_DIR = Ordner mit echten Fotos

Das teure ML-Modell wird hier NICHT geladen.
"""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aicheck.scoring import aggregate, GREEN, AMBER, RED  # noqa: E402
from aicheck.signals.base import SignalResult  # noqa: E402
from aicheck.signals.ml_local import _ai_probability  # noqa: E402
from aicheck.signals.c2pa import C2paSignal  # noqa: E402
from aicheck.signals.metadata import MetadataSignal  # noqa: E402
from aicheck.signals.forensics import ForensicsSignal  # noqa: E402
from aicheck.signals.online import _parse_sightengine  # noqa: E402

AI_DIR = (
    Path(os.environ["AICHECK_TEST_AI_DIR"])
    if os.environ.get("AICHECK_TEST_AI_DIR")
    else None
)
REAL_DIR = (
    Path(os.environ["AICHECK_TEST_REAL_DIR"])
    if os.environ.get("AICHECK_TEST_REAL_DIR")
    else None
)


def _sig(score, weight=1.0, hard=None):
    return SignalResult("t", score, weight, "d", hard_verdict=hard)


def _first(dir_, *exts):
    if dir_ is None or not dir_.is_dir():
        return None
    for f in sorted(dir_.glob("*")):
        if f.is_file() and not f.name.startswith("thumb_") and f.suffix.lower() in exts:
            return f
    return None


class TestScoring(unittest.TestCase):
    def test_empty_signals_is_amber(self):
        v = aggregate([])
        self.assertEqual(v.level, AMBER)
        self.assertIsNone(v.score)

    def test_all_na_is_amber(self):
        v = aggregate([_sig(None), _sig(None)])
        self.assertEqual(v.level, AMBER)

    def test_high_score_is_red(self):
        self.assertEqual(aggregate([_sig(0.9)]).level, RED)

    def test_low_score_is_green(self):
        self.assertEqual(aggregate([_sig(0.1)]).level, GREEN)

    def test_mid_score_is_amber(self):
        self.assertEqual(aggregate([_sig(0.45)]).level, AMBER)

    def test_weighting(self):
        # starkes echtes Signal (Gewicht 3) überstimmt schwaches KI-Signal
        v = aggregate([_sig(0.1, weight=3.0), _sig(0.9, weight=0.5)])
        self.assertEqual(v.level, GREEN)

    def test_na_excluded_from_mean(self):
        v = aggregate([_sig(None, weight=5.0), _sig(0.9, weight=1.0)])
        self.assertEqual(v.level, RED)

    def test_hard_ai_override_beats_score(self):
        # niedriger Score, aber hartes KI-Verdikt -> RED
        v = aggregate([_sig(0.0, weight=10.0), _sig(0.99, hard="ai")])
        self.assertEqual(v.level, RED)
        self.assertEqual(v.score, 1.0)

    def test_hard_real_override_beats_score(self):
        v = aggregate([_sig(0.99, weight=10.0), _sig(0.0, hard="real")])
        self.assertEqual(v.level, GREEN)

    def test_custom_thresholds(self):
        self.assertEqual(aggregate([_sig(0.5)], red_threshold=0.5).level, RED)


class TestLabelMapping(unittest.TestCase):
    def test_fake_real_labels(self):
        preds = [{"label": "fake", "score": 0.8}, {"label": "real", "score": 0.2}]
        self.assertAlmostEqual(_ai_probability(preds), 0.8)

    def test_artificial_human_labels(self):
        preds = [
            {"label": "artificial", "score": 0.3},
            {"label": "human", "score": 0.7},
        ]
        self.assertAlmostEqual(_ai_probability(preds), 0.3)

    def test_unmappable_labels_return_none(self):
        preds = [{"label": "LABEL_0", "score": 0.6}, {"label": "LABEL_1", "score": 0.4}]
        self.assertIsNone(_ai_probability(preds))


class TestSightengineParsing(unittest.TestCase):
    def test_success_with_generator_breakdown(self):
        data = {
            "status": "success",
            "type": {
                "ai_generated": 0.99,
                "ai_generators": {"midjourney": 0.81, "flux": 0.10},
            },
        }
        score, detail = _parse_sightengine(data)
        self.assertAlmostEqual(score, 0.99)
        self.assertIn("midjourney", detail)

    def test_failure_status_returns_none(self):
        data = {"status": "failure", "error": {"message": "invalid credentials"}}
        score, detail = _parse_sightengine(data)
        self.assertIsNone(score)
        self.assertIn("invalid credentials", detail)

    def test_missing_field_returns_none(self):
        score, detail = _parse_sightengine({"status": "success", "type": {}})
        self.assertIsNone(score)


class TestCache(unittest.TestCase):
    def test_image_hash_is_content_based(self):
        import tempfile
        from aicheck.cache import image_hash

        with tempfile.TemporaryDirectory() as d:
            a = Path(d) / "a.bin"
            a.write_bytes(b"hello")
            b = Path(d) / "b.bin"
            b.write_bytes(b"hello")
            c = Path(d) / "c.bin"
            c.write_bytes(b"world")
            self.assertEqual(
                image_hash(a), image_hash(b)
            )  # gleicher Inhalt -> gleicher Hash
            self.assertNotEqual(
                image_hash(a), image_hash(c)
            )  # anderer Inhalt -> anderer Hash

    def test_put_get_roundtrip_and_clear(self):
        import tempfile
        from aicheck.cache import OnlineCache, clear

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cache.json"
            c = OnlineCache(p)
            self.assertIsNone(c.get("sightengine", "abc"))
            c.put("sightengine", "abc", {"score": 0.9, "detail": "x"})
            # neue Instanz liest die persistierten Daten
            self.assertEqual(OnlineCache(p).get("sightengine", "abc")["score"], 0.9)
            self.assertEqual(clear(p), 1)
            self.assertIsNone(OnlineCache(p).get("sightengine", "abc"))


class TestSignalsOnImages(unittest.TestCase):
    def test_c2pa_no_manifest_is_na(self):
        f = _first(AI_DIR, ".png") or _first(REAL_DIR, ".jpg", ".png")
        if not f:
            self.skipTest("keine Testbilder vorhanden")
        r = C2paSignal().evaluate(f)
        self.assertIsNone(r.score)  # gestripte Bilder -> kein Urteil

    def test_metadata_ai_dim_more_suspicious_than_real(self):
        ai = _first(AI_DIR, ".png")
        real = _first(REAL_DIR, ".jpg", ".png")
        if not (ai and real):
            self.skipTest("keine Vergleichsbilder vorhanden")
        s_ai = MetadataSignal().evaluate(ai).score
        s_real = MetadataSignal().evaluate(real).score
        self.assertIsNotNone(s_ai)
        self.assertIsNotNone(s_real)
        # KI-typische Auflösung soll nicht weniger verdächtig sein als echtes Foto
        self.assertGreaterEqual(s_ai, s_real)

    def test_forensics_returns_bounded_score(self):
        f = _first(AI_DIR, ".png") or _first(REAL_DIR, ".jpg", ".png")
        if not f:
            self.skipTest("keine Testbilder vorhanden")
        r = ForensicsSignal().evaluate(f)
        if r.score is not None:
            self.assertGreaterEqual(r.score, 0.0)
            self.assertLessEqual(r.score, 1.0)


if __name__ == "__main__":
    unittest.main()
