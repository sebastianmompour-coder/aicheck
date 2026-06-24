"""CLI für aicheck — Verdachts-Triage, ob Bilder KI-generiert sind."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .scoring import GREEN, AMBER, RED, Verdict, aggregate
from .signals.base import SignalResult
from .signals.c2pa import C2paSignal
from .signals.forensics import ForensicsSignal
from .signals.metadata import MetadataSignal
from .signals.ml_local import DEFAULT_MODEL, MlLocalSignal
from .signals.online import OnlineSignal

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}

# Kuratierte Detektormodelle. Alias -> (HuggingFace-ID, Kurzbeschreibung).
# Jede beliebige HF-"image-classification"-Modell-ID ist ebenfalls erlaubt.
# Messwerte aus tests/eval_model.py (echte Referenzfotos vs. Verdachtsbilder).
MODEL_CATALOG: list[tuple[str, str, str]] = [
    (
        "haywood",
        "haywoodsloan/ai-image-detector-deploy",
        "DEFAULT. Beste Werte im Test: echt 0.13 / KI 0.99, FP 1/8, keine verpassten KI-Bilder.",
    ),
    (
        "sdxl",
        "Organika/sdxl-detector",
        "Sehr sensitiv: echt 0.24 / KI 1.00, FP 2/8. Gut wenn nichts durchrutschen darf.",
    ),
    (
        "ateeqq",
        "Ateeqq/ai-vs-human-image-detector",
        "Schwächer im Test: FP 2/8 UND verpasste 1 KI-Bild (FN).",
    ),
    (
        "umm",
        "umm-maybe/AI-image-detector",
        "Älteres Basismodell (hier ungetestet), schwächer bei neuen Generatoren.",
    ),
]
_ALIASES = {alias: hf_id for alias, hf_id, _ in MODEL_CATALOG}


def resolve_model(name: str) -> str:
    """Alias -> HF-ID; unbekannte Namen werden als HF-ID durchgereicht."""
    return _ALIASES.get(name, name)


def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


_LEVEL_COLOR = {RED: "31", AMBER: "33", GREEN: "32"}


def _c(text: str, code: str, on: bool) -> str:
    return f"\033[{code}m{text}\033[0m" if on else text


def _fmt_score(s: float | None) -> str:
    return f"{s:.2f}" if s is not None else " – "


def _bar(score: float | None, width: int = 12) -> str:
    if score is None:
        return " " * width
    filled = max(0, min(width, round(score * width)))
    return "█" * filled + "░" * (width - filled)


#  Orchestrierung
def build_signals(
    model: str, online: bool, with_forensics: bool, with_ml: bool, refresh: bool = False
) -> list:
    signals = [C2paSignal(), MetadataSignal()]
    if with_forensics:
        signals.append(ForensicsSignal())
    if with_ml:
        signals.append(MlLocalSignal(model=model))
    signals.append(OnlineSignal(enabled=online, refresh=refresh))
    return signals


def analyze(path: Path, signals: list, amber: float, red: float) -> Verdict:
    results: list[SignalResult] = [s.evaluate(path) for s in signals]
    return aggregate(results, amber_threshold=amber, red_threshold=red)


def collect_images(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            out.extend(
                sorted(
                    f
                    for f in path.rglob("*")
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTS
                )
            )
        elif path.is_file():
            out.append(path)
        else:
            print(f"WARN: Pfad nicht gefunden: {p}", file=sys.stderr)
    return out


#  Ausgaben
def print_report(path: Path, v: Verdict, color: bool, index: str = "") -> None:
    cc = _LEVEL_COLOR[v.level]
    print(_c("─" * 74, "90", color))
    prefix = f"{index}  " if index else ""
    head = f"{prefix}{v.emoji}  {_c(v.label.upper(), cc, color)}"
    print(
        f"{head}    Score {_c(_fmt_score(v.score), cc, color)}  "
        f"[{_c(_bar(v.score), cc, color)}]"
    )
    print(f"   {_c(path.name, '1', color)}")
    if str(path.parent) not in (".", ""):
        print(f"   {_c(str(path.parent) + '/', '90', color)}")
    print(f"   Grund: {v.reason}")
    print()
    header = f"{'Signal':<11}{'Score':>6}   Bewertung"
    print(f"   {_c(header, '90', color)}")
    for s in v.signals:
        mark = _c("!", "31", color) if s.error else " "
        print(f"   {mark}{s.name:<10}{_fmt_score(s.score):>6}   {s.detail}")
    print()


def _counts(rows: list[tuple[Path, Verdict]]) -> dict:
    c = {GREEN: 0, AMBER: 0, RED: 0}
    for _, v in rows:
        c[v.level] += 1
    return c


def print_summary(rows: list[tuple[Path, Verdict]], color: bool) -> None:
    c = _counts(rows)
    print(_c("═" * 74, "90", color))
    print(
        f"Zusammenfassung: {len(rows)} Bilder     "
        f"{_c('🔴 ' + str(c[RED]), '31', color)}    "
        f"{_c('🟡 ' + str(c[AMBER]), '33', color)}    "
        f"{_c('🟢 ' + str(c[GREEN]), '32', color)}"
    )
    print(_c("═" * 74, "90", color))


def print_table(rows: list[tuple[Path, Verdict]]) -> None:
    print(f"\n{'Ampel':<6} {'Score':>5}  {'Datei':<34} Hauptgrund")
    print("-" * 118)
    for path, v in rows:
        name = path.name if len(path.name) <= 34 else path.name[:31] + "..."
        reason = v.reason if len(v.reason) <= 66 else v.reason[:63] + "..."
        print(f"{v.emoji:<5} {_fmt_score(v.score):>5}  {name:<34} {reason}")
    print("-" * 118)
    c = _counts(rows)
    print(
        f"Summe: 🔴 {c[RED]}   🟡 {c[AMBER]}   🟢 {c[GREEN]}   (von {len(rows)} Bildern)"
    )


def to_json(rows: list[tuple[Path, Verdict]]) -> str:
    payload = [
        {
            "file": str(path),
            "level": v.level,
            "score": v.score,
            "reason": v.reason,
            "signals": [s.as_dict() for s in v.signals],
        }
        for path, v in rows
    ]
    return json.dumps(payload, indent=2, ensure_ascii=False)


def print_model_list() -> None:
    print("Bekannte Detektormodelle (Alias  ->  HuggingFace-ID):\n")
    for alias, hf_id, desc in MODEL_CATALOG:
        print(f"  {alias:<9} {hf_id}")
        print(f"  {'':<9} {desc}\n")
    print("Nutzung:  --model <alias>   oder   --model <beliebige-hf-id>")
    print("Vergleich auf eigenen Daten:  .venv/bin/python tests/eval_model.py <id>")


# ---------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aicheck",
        description="Triage: schätzt ein, ob Bilder KI-generiert sind. "
        "Liefert einen begründeten Verdacht, keinen Beweis.",
    )
    parser.add_argument("paths", nargs="*", help="Bilddatei(en) oder Ordner")
    parser.add_argument(
        "--online",
        action="store_true",
        help="optionalen Online-Detektor zuschalten (lädt Bild hoch)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Online-Cache für diese Bilder ignorieren und neu abfragen",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Online-Ergebnis-Cache leeren und beenden",
    )
    parser.add_argument(
        "--json", action="store_true", help="JSON statt Report ausgeben"
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="kompakte Tabelle statt ausführlicher Reports",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Detektormodell: Alias (siehe --list-models) oder HF-ID",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="bekannte Detektormodelle auflisten und beenden",
    )
    parser.add_argument(
        "--no-ml",
        action="store_true",
        help="ML-Signal weglassen (kein Modell-Download/Inferenz)",
    )
    parser.add_argument(
        "--no-forensics", action="store_true", help="Frequenz-Forensik-Signal weglassen"
    )
    parser.add_argument(
        "--amber",
        type=float,
        default=0.35,
        help="Score-Schwelle ab der 🟡 (Default 0.35)",
    )
    parser.add_argument(
        "--red",
        type=float,
        default=0.60,
        help="Score-Schwelle ab der 🔴 (Default 0.60)",
    )
    args = parser.parse_args(argv)

    if args.list_models:
        print_model_list()
        return 0
    if args.clear_cache:
        from .cache import clear

        print(f"Online-Cache geleert ({clear()} Einträge entfernt).")
        return 0
    if not args.paths:
        parser.error(
            "keine Pfade angegeben (oder --list-models / --clear-cache verwenden)"
        )

    images = collect_images(args.paths)
    if not images:
        print("Keine Bilder gefunden.", file=sys.stderr)
        return 2

    signals = build_signals(
        model=resolve_model(args.model),
        online=args.online,
        with_forensics=not args.no_forensics,
        with_ml=not args.no_ml,
        refresh=args.refresh,
    )

    rows: list[tuple[Path, Verdict]] = []
    for img in images:
        rows.append((img, analyze(img, signals, args.amber, args.red)))

    if args.json:
        print(to_json(rows))
    elif args.table:
        print_table(rows)
    else:
        color = _supports_color()
        total = len(rows)
        for i, (path, v) in enumerate(rows, 1):
            idx = f"[{i}/{total}]" if total > 1 else ""
            print_report(path, v, color, index=idx)
        if total > 1:
            print_summary(rows, color)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
