"""Tests für aicheck. Läuft ohne pytest:  .venv/bin/python -m unittest -v

Kernlogik (Scoring, Label-Mapping) wird synthetisch getestet (schnell, kein I/O).
Die optionalen Signal-Smoketests laufen gegen eigene Bildordner und werden
übersprungen, falls die Ordner fehlen. Pfade über Umgebungsvariablen setzen:

  AICHECK_TEST_AI_DIR   = Ordner mit (vermutlich) KI-Bildern
  AICHECK_TEST_REAL_DIR = Ordner mit echten Fotos

Das teure ML-Modell wird hier NICHT geladen.
"""

import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aicheck.scoring import aggregate, GREEN, AMBER, RED  # noqa: E402
from aicheck.signals.base import SignalResult, find_keyword  # noqa: E402
from aicheck.signals.ml_local import _ai_probability  # noqa: E402
from aicheck.signals.c2pa import C2paSignal, assess_manifest  # noqa: E402
from aicheck.signals.metadata import MetadataSignal, assess_exif  # noqa: E402
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


class TestKeywordMatching(unittest.TestCase):
    def test_matches_whole_words_only(self):
        self.assertEqual(find_keyword("Made with FLUX.1 dev", ("flux",)), "flux")
        self.assertIsNone(find_keyword("influx of tourists", ("flux",)))
        self.assertIsNone(find_keyword("Dallas skyline", ("dall-e", "dalle")))
        self.assertEqual(find_keyword("DALL-E 3", ("dall-e",)), "dall-e")

    def test_allows_punctuation_boundaries(self):
        self.assertEqual(find_keyword("ComfyUI/0.3", ("comfyui",)), "comfyui")
        self.assertEqual(find_keyword("stable-diffusion-xl", ("stable-diffusion",)), "stable-diffusion")


def _exif_camera(**extra):
    base = {
        "SourceFile": "/tmp/IMG_0001.jpg",
        "File:FileName": "IMG_0001.jpg",
        "File:Directory": "/tmp",
        "EXIF:Make": "Apple",
        "EXIF:Model": "iPhone 15",
        "EXIF:ExposureTime": "1/120",
        "EXIF:FNumber": 1.8,
        "EXIF:ISO": 50,
        "EXIF:DateTimeOriginal": "2026:01:01 10:00:00",
        "File:ImageWidth": 4032,
        "File:ImageHeight": 3024,
    }
    base.update(extra)
    return base


class TestMetadataAssessment(unittest.TestCase):
    """Reine Bewertung einer exiftool-JSON-Ausgabe, ohne exiftool."""

    def test_camera_exif_is_real(self):
        r = assess_exif(_exif_camera())
        self.assertEqual(r.score, 0.1)
        self.assertIn("Kamera-EXIF", r.detail)

    def test_path_never_counts_as_generator_hint(self):
        # Regression: Ordner "gemini/" machte jedes Bild darin KI-verdächtig.
        data = _exif_camera(
            SourceFile="/home/x/Downloads/gemini/flux/IMG_0001.jpg",
            **{"File:Directory": "/home/x/Downloads/gemini/flux",
               "File:FileName": "midjourney-test.jpg"},
        )
        self.assertEqual(assess_exif(data).score, 0.1)

    def test_tag_names_are_not_searched_as_text(self):
        # Ein Tag-NAME wie "XMP:Gemini..." darf nicht als Generator-Wert zählen.
        data = _exif_camera(**{"XMP:GeminiRating": 5})
        self.assertEqual(assess_exif(data).score, 0.1)

    def test_jpeg_comment_from_gd_is_not_ai(self):
        # Regression: PHP/GD schreibt ein COM-Segment -> "File:Comment".
        data = _exif_camera(
            **{"File:Comment": "CREATOR: gd-jpeg v1.0 (using IJG JPEG v80), quality = 90"}
        )
        self.assertEqual(assess_exif(data).score, 0.1)

    def test_generator_in_software_tag(self):
        data = {"SourceFile": "/tmp/a.png", "PNG:Software": "NovelAI", "File:ImageWidth": 832}
        r = assess_exif(data)
        self.assertEqual(r.score, 0.92)
        self.assertIn("novelai", r.detail)

    def test_generator_substring_inside_other_word_is_ignored(self):
        data = {"SourceFile": "/tmp/a.jpg", "XMP:Description": "Influx of visitors in Dallas"}
        self.assertLess(assess_exif(data).score, 0.9)

    def test_comfyui_workflow_chunk(self):
        data = {"SourceFile": "/tmp/a.png", "PNG:Workflow": '{"nodes": []}', "PNG:Prompt": "{}"}
        r = assess_exif(data)
        self.assertEqual(r.score, 0.85)
        self.assertIn("workflow", r.detail)

    def test_a1111_parameters_in_user_comment(self):
        # A1111 schreibt bei JPEG den Parameterstring in EXIF:UserComment.
        data = {
            "SourceFile": "/tmp/a.jpg",
            "EXIF:UserComment": "a cat\nNegative prompt: blurry\nSteps: 20, Sampler: Euler a, CFG scale: 7",
        }
        r = assess_exif(data)
        self.assertEqual(r.score, 0.85)
        self.assertIn("Parameterstring", r.detail)

    def test_stripped_with_ai_dims(self):
        data = {"SourceFile": "/tmp/a.png", "File:ImageWidth": 1536, "File:ImageHeight": 1024}
        self.assertEqual(assess_exif(data).score, 0.65)

    def test_stripped_with_other_dims(self):
        data = {"SourceFile": "/tmp/a.png", "File:ImageWidth": 1500, "File:ImageHeight": 1000}
        self.assertEqual(assess_exif(data).score, 0.55)

    def test_evaluate_uses_assess(self):
        # Subprocess gemockt: evaluate() muss die reine Bewertung durchreichen.
        import subprocess
        from unittest import mock

        proc = subprocess.CompletedProcess([], 0, stdout=json.dumps([_exif_camera()]), stderr="")
        with mock.patch("shutil.which", return_value="/usr/bin/exiftool"), mock.patch(
            "subprocess.run", return_value=proc
        ) as run:
            r = MetadataSignal().evaluate(Path("IMG_0001.jpg"))
        self.assertEqual(r.score, 0.1)
        self.assertTrue(Path(run.call_args[0][0][-1]).is_absolute())


def _manifest(title="photo.jpg", generator="Leica M11-P", source_type=None, actions=None):
    assertions = []
    if source_type:
        assertions.append(
            {
                "label": "stds.iptc.photo-metadata",
                "data": {
                    "Iptc4xmpExt:DigitalSourceType": f"http://cv.iptc.org/newscodes/digitalsourcetype/{source_type}"
                },
            }
        )
    if actions:
        assertions.append({"label": "c2pa.actions", "data": {"actions": actions}})
    return {
        "active_manifest": "urn:x",
        "manifests": {
            "urn:x": {
                "title": title,
                "claim_generator": generator,
                "claim_generator_info": [{"name": generator, "version": "1.0"}],
                "signature_info": {"issuer": "Leonardo Rossi Photography"},
                "assertions": assertions,
            }
        },
    }


class TestC2paAssessment(unittest.TestCase):
    """Reine Bewertung einer c2patool-JSON-Ausgabe, ohne c2patool."""

    def test_digital_capture_is_hard_real(self):
        r = assess_manifest(_manifest(source_type="digitalCapture"))
        self.assertEqual(r.hard_verdict, "real")

    def test_trained_algorithmic_media_is_hard_ai(self):
        r = assess_manifest(
            _manifest(generator="ChatGPT", source_type="trainedAlgorithmicMedia")
        )
        self.assertEqual(r.hard_verdict, "ai")
        self.assertEqual(aggregate([r]).level, RED)

    def test_generator_name_in_claim_generator(self):
        r = assess_manifest(_manifest(generator="OpenAI-API"))
        self.assertEqual(r.hard_verdict, "ai")
        self.assertIn("openai", r.detail)

    def test_generator_name_in_software_agent(self):
        r = assess_manifest(
            _manifest(actions=[{"action": "c2pa.created", "softwareAgent": {"name": "Adobe Firefly"}}])
        )
        self.assertEqual(r.hard_verdict, "ai")
        self.assertIn("firefly", r.detail)

    def test_title_and_issuer_never_trigger_ai(self):
        # Regression: Dateiname "flux-festival.jpg" + Issuer "Leonardo ..." ->
        # wurde als KI-Generator gewertet und überstimmte das Kamera-Manifest.
        r = assess_manifest(_manifest(title="flux-festival.jpg", source_type="digitalCapture"))
        self.assertEqual(r.hard_verdict, "real")
        self.assertEqual(aggregate([r]).level, GREEN)

    def test_unknown_provenance_is_soft_neutral(self):
        r = assess_manifest(_manifest(generator="Adobe Photoshop 26.0"))
        self.assertEqual(r.score, 0.5)
        self.assertIsNone(r.hard_verdict)


class TestVerdictLabels(unittest.TestCase):
    def test_labels_are_proper_utf8(self):
        self.assertEqual(aggregate([_sig(0.9)]).label, "KI-verdächtig")
        self.assertEqual(aggregate([_sig(0.1)]).label, "echt-wahrscheinlich")


if __name__ == "__main__":
    unittest.main()
