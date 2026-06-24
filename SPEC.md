# aicheck — KI-Bild-Erkennung (Design-Spec)

**Datum:** 2026-06-24
**Status:** Design freigegeben, Implementierung ausstehend

## Ziel

Ein lokales CLI-Tool, das für ein Bild oder einen Ordner einschätzt, ob es sich
um ein KI-generiertes Bild handelt. Zweck: **Verdachts-Triage** — schnelle
Vorsortierung, welche Bilder eine genauere manuelle Prüfung verdienen. Recall
wichtiger als Präzision (im Zweifel eher als verdächtig markieren).

### Bewusst KEIN Ziel (YAGNI / Ehrlichkeit)

- Kein „100 % sicher KI ja/nein". Das ist technisch unmöglich; jeder Detektor ist
  probabilistisch und durch Re-Encode/Screenshot/neues Modell täuschbar.
- Keine GUI, kein Web-Service. Reines CLI.
- Keine eigene Modell-Trainierung. Wir nutzen ein vorhandenes Modell.

## Kontext (Ausgangslage)

- Beispieldaten: ein Satz von 7 Verdachtsbildern, alle 1536×1024, ~2–2,6 MB, RGB+Alpha.
- exiftool nutzlos, weil Bilder **komplett metadaten-gestript** sind: keine
  Kamera-EXIF, kein Software-Tag, **kein C2PA-Manifest**.
- Umgebung: macOS, Homebrew, `exiftool` 13.55, `ollama` (Text-Modelle),
  Python 3.14 (Risiko: evtl. keine `torch`-Wheels → ONNX-Fallback).

## Architektur: Multi-Signal-Ampel

Pro Bild werden mehrere **unabhängige Signale** gesammelt und zu einer Ampel
verrechnet: 🟢 echt-wahrscheinlich / 🟡 unklar / 🔴 KI-verdächtig.

Jedes Signal ist ein eigenständiges, einzeln testbares Modul mit einheitlicher
Schnittstelle:

```
Signal.evaluate(image_path) -> SignalResult(
    name: str,
    score: float | None,   # 0.0 = real ... 1.0 = KI; None = nicht anwendbar
    weight: float,          # Gewicht im Gesamtscore
    detail: str,            # menschenlesbare Begründung
)
```

### Signale

| Modul | Was es prüft | Stärke | Default an |
|---|---|---|---|
| `c2pa` | `c2patool` liest Content-Credentials-Manifest (OpenAI/Adobe/Google) | hart, aber meist gestript | ja |
| `metadata` | fehlende EXIF, verdächtige Auflösung (1024er-Vielfache), Software-Tags, PNG-Profile | schwach, billig | ja |
| `forensics` | leichte Frequenz-/Noise-Heuristik (FFT-Residuen), ohne ML | schwach, erklärbar | ja |
| `ml_local` | dedizierter HuggingFace-Klassifikator „AI vs. real", **swappbar via Config** | mittel, Hauptsignal | ja |
| `online` | generischer Provider-Adapter (Stub), nur bei `--online` oder 🟡 | mittel-stark, Blackbox | nein |

### Scoring

- Gesamtscore = gewichteter Mittelwert der verfügbaren Signal-Scores
  (Signale mit `score=None` zählen nicht mit).
- Schwellen (konfigurierbar): `< 0.35` → 🟢, `0.35–0.6` → 🟡, `> 0.6` → 🔴.
- **Hartes Override:** valides C2PA-Manifest eines Generators → sofort 🔴 mit
  Begründung; C2PA „echte Kamera"-Beleg → starkes 🟢-Signal.
- Triage-Bias: bei genau einem fehlenden/ambivalenten Hauptsignal eher 🟡 als 🟢.

## CLI

```
aicheck PFAD [PFAD ...] [--online] [--json] [--model NAME] [--threshold-amber X --threshold-red Y]
```

- **Einzeldatei** → Detail-Report mit allen Einzelsignalen + Gesamt-Ampel.
- **Ordner** → Tabelle (Datei → Ampel + Score + Hauptgrund) + Zusammenfassung
  (Anzahl 🟢/🟡/🔴).
- `--json` für maschinenlesbare Ausgabe (Weiterverarbeitung/Logging).
- `--online` schaltet den Online-Adapter zu; **ohne Flag werden nie Daten
  hochgeladen**.

## ML-Komponente (Option A, freigegeben)

- Dedizierter HuggingFace-Bildklassifikator als Hauptsignal.
- **Swappbar** über Config/`--model`; konkretes Modell wird im
  Implementierungsschritt gegen aktuelle Verfügbarkeit/Qualität geprüft und
  ausgewählt (Kandidaten: SwinV2-/ViT-basierte AI-Detektoren).
- Lazy-Load: Modell wird erst beim ersten ML-Signal geladen, einmal pro Lauf.

## Online-Komponente (Adapter-Stub, freigegeben)

- Generische `OnlineProvider`-Schnittstelle (`detect(image) -> score, detail`).
- Mitgelieferter Stub/Platzhalter; konkreter Dienst + API-Key später per `.env`.
- Kein Lock-in auf einen bestimmten Anbieter.

## Setup / Dependency-Strategie

- Isoliertes **venv** im `aicheck/`-Ordner (nicht das System-Python belasten).
- ML-Inferenz bevorzugt über **ONNX Runtime**, um die `torch`/Python-3.14-Wheel-
  Frage zu umgehen; `transformers`+`torch` nur falls für das gewählte Modell nötig.
- `c2patool` via Homebrew/Cargo (Install-Check + klare Fehlermeldung, wenn fehlt).
- `requirements.txt` + kurzes `setup.sh`/README für reproduzierbares Setup.

## Fehlerverhalten

- Jedes Signal kapselt seine Fehler: schlägt ein Signal fehl (z.B. `c2patool`
  nicht installiert, Modell nicht ladbar), liefert es `score=None` + Detail-
  Hinweis statt das ganze Tool abzubrechen.
- Nicht lesbare/ungültige Bilddatei → klarer Fehler pro Datei, Batch läuft weiter.
- Exit-Code: 0 bei Erfolg; non-zero nur bei echten Tool-Fehlern, nicht bei „🔴".

## Teststrategie

- Unit-Tests pro Signal-Modul mit Fixtures (ein bekannt KI-Bild, ein bekannt
  echtes Foto mit EXIF, ein gestriptes PNG).
- Scoring-Aggregator getestet mit synthetischen SignalResults (Schwellen,
  None-Handling, C2PA-Override).
- CLI-Smoke-Test gegen einen Beispielordner.

## Bekannte Grenzen (im README dokumentieren)

- Neue Generatoren (GPT-Image, Flux, Midjourney v6) werden von älteren
  Detektoren schlecht erkannt → Modell muss tauschbar bleiben.
- Re-Encodes/Screenshots/Postprocessing senken die Trefferquote.
- Ergebnis ist ein **Verdacht**, kein Beweis. Entscheidung bleibt beim Menschen.
