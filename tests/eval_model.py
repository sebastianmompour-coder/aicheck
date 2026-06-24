"""Ad-hoc-Eval: vergleicht ein Detektormodell auf zwei eigenen Bildordnern
(z.B. bekannt echt vs. verdächtig) und zeigt P(KI) je Gruppe.

Aufruf:
  .venv/bin/python tests/eval_model.py <modell-alias-oder-hf-id> <ordner_echt> <ordner_verdacht>

Beispiel:
  .venv/bin/python tests/eval_model.py haywood ./samples/echt ./samples/verdacht
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aicheck.cli import resolve_model  # noqa: E402
from aicheck.signals.ml_local import MlLocalSignal  # noqa: E402

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
THRESH = 0.6


def images_in(folder: str) -> list[Path]:
    base = Path(folder)
    return [
        f
        for f in sorted(base.rglob("*"))
        if f.is_file()
        and not f.name.startswith("thumb_")
        and f.suffix.lower() in IMG_EXTS
    ]


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    model, real_dir, suspect_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    sig = MlLocalSignal(model=resolve_model(model))
    groups = {
        f"ECHT     {real_dir}": images_in(real_dir),
        f"VERDACHT {suspect_dir}": images_in(suspect_dir),
    }
    print(f"\n=== Modell: {resolve_model(model)} ===")
    for label, files in groups.items():
        scores = [
            r.score for r in (sig.evaluate(f) for f in files) if r.score is not None
        ]
        if not scores:
            print(f"{label}: keine Bilder/Scores")
            continue
        n = len(scores)
        over = sum(1 for s in scores if s >= THRESH)
        print(
            f"{label:<28} n={n:<3} P(KI) mean={sum(scores) / n:.3f} "
            f"min={min(scores):.3f} max={max(scores):.3f}  >={THRESH}: {over}/{n}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
