# aicheck — KI-Bild-Triage

Schätzt für ein Bild oder einen Ordner ein, ob es **KI-generiert** ist. Zweck ist
**Verdachts-Triage**: schnell vorsortieren, welche Bilder eine genauere manuelle
Prüfung verdienen.

> ⚠️ **Kein Beweis, nur ein Verdacht.** Es gibt kein Tool, das zuverlässig zu
> 100 % „KI ja/nein" sagt. Jeder Detektor ist probabilistisch und durch
> Re-Encode, Screenshot oder neue Generatoren täuschbar. aicheck liefert eine
> begründete Ampel — die Entscheidung bleibt beim Menschen.

## Funktionsweise

Pro Bild werden mehrere **unabhängige Signale** gesammelt und zu einer Ampel
verrechnet — 🟢 echt-wahrscheinlich / 🟡 unklar / 🔴 KI-verdächtig:

| Signal | prüft | Stärke |
|---|---|---|
| `c2pa` | C2PA-/Content-Credentials-Manifest (IPTC `digitalSourceType`, Generator) via `c2patool` | **hart**, aber meist gestript |
| `metadata` | fehlende Kamera-EXIF, KI-typische Auflösung, Generator-Tags (ComfyUI/A1111/…) via `exiftool` | schwach, billig |
| `forensics` | Hochfrequenz-Energie im FFT-Spektrum (numpy) | schwach, erklärbar |
| `ml_local` | HuggingFace-Detektor „AI vs. real" (**Hauptsignal**, swappbar) | mittel |
| `online` | optionaler externer Dienst (Adapter-Stub), nur mit `--online` | mittel-stark |

Gesamtscore = gewichteter Mittelwert aller anwendbaren Signale. Ein valides
C2PA-Manifest **überstimmt** den Score (hartes Verdikt). Score `0.0` = echt …
`1.0` = KI.

## Setup

Läuft auf **macOS, Linux, Windows und WSL** (Python ≥ 3.11). Das Tool selbst ist
plattformneutral; nur die Installation der externen Tools unterscheidet sich.

### 1. Python-Umgebung

**macOS / Linux / WSL** (Komfort-Script):

```bash
cd aicheck
./setup.sh
```

**Alle Plattformen manuell** (auch Windows):

```bash
python -m venv .venv
# venv aktivieren:
#   macOS/Linux/WSL :  source .venv/bin/activate
#   Windows (PowerShell):  .venv\Scripts\Activate.ps1
#   Windows (cmd)   :  .venv\Scripts\activate.bat
pip install -e .
```

Danach gibt es den plattformneutralen Befehl **`aicheck`** (im aktivierten venv).

### 2. Externe Tools (optional)

`exiftool` und `c2patool` verbessern die Erkennung, sind aber **optional** —
fehlen sie, liefert das jeweilige Signal „n/a" statt eines Fehlers.

| Tool | macOS | Linux / WSL | Windows |
|---|---|---|---|
| exiftool | `brew install exiftool` | `apt install libimage-exiftool-perl` | [exiftool.org](https://exiftool.org) (Exe) · `winget install OliverBetz.ExifTool` |
| c2patool | `brew install c2patool` | `cargo install c2patool` | [c2patool-Releases](https://github.com/contentauth/c2patool/releases) · `cargo install c2patool` |

Auf Windows die heruntergeladenen `.exe` in einen Ordner im `PATH` legen.

Getestet mit Python 3.14 / torch 2.12 / transformers 5.12 (macOS arm64);
funktioniert auch mit Python 3.11–3.13. Das Tool **bringt keine Bilder mit** —
teste mit eigenen. Beim ersten Lauf lädt das ML-Modell einmalig herunter
(~hunderte MB), danach läuft alles offline.

### Datenschutz / offline

Die **Bildanalyse läuft vollständig lokal** auf der CPU — ohne `--online`
verlässt **kein Bild** den Rechner. Das einzige, was Netz braucht, ist der
**einmalige Modell-Download** vom HuggingFace-Hub. Liegt das Modell danach im
Cache (`~/.cache/huggingface`), schaltet aicheck automatisch auf
`HF_HUB_OFFLINE` und macht **gar keine** Hub-Anfragen mehr (auch keinen
Versions-Check) — daher erscheint die „unauthenticated requests"-Warnung nur
beim allerersten Download. Verifizierbar: mit unerreichbarem Hub
(`HF_ENDPOINT=http://127.0.0.1:9 ./aicheck.sh bild.png`) läuft es trotzdem durch.

## Nutzung

Nach dem Setup im aktivierten venv (alle Plattformen gleich):

```bash
aicheck bild.png                    # Einzeldatei -> ausführlicher Report
aicheck /pfad/zu/bildern            # Ordner -> ein Report je Bild + Zusammenfassung
aicheck /pfad/zu/bildern --table    # kompakte Tabelle statt Reports
aicheck /pfad/zu/bildern --json     # maschinenlesbar
aicheck --list-models               # verfügbare Detektormodelle anzeigen
aicheck bild.png --model sdxl       # anderes Modell (Alias oder HF-ID)
aicheck bild.png --no-ml            # nur C2PA + Metadaten + Forensik (kein Modell)
aicheck bild.png --online           # zusätzlich Online-Detektor (lädt Bild hoch)
aicheck bild.png --online --refresh # Online-Cache ignorieren, neu abfragen
aicheck --clear-cache               # Online-Ergebnis-Cache leeren
aicheck /pfad/zu/bildern --red 0.5  # aggressivere Schwelle (mehr 🔴, mehr Recall)
```

Ohne venv-Aktivierung geht es überall via `python -m aicheck …` (z.B.
`.venv/bin/python -m aicheck …` bzw. `.venv\Scripts\python -m aicheck …`).
macOS/Linux haben zusätzlich den Shortcut `./aicheck.sh …`.

Standardausgabe ist der **ausführliche Report** (Einzelbild *und* Ordner). Flags:
`--table` (kompakte Übersicht), `--json`, `--amber`/`--red` (Ampel-Schwellen,
Default 0.35 / 0.60), `--model`, `--no-forensics`, `--no-ml`, `--list-models`.

### Modelle

`--list-models` zeigt die kuratierten Detektoren mit Messwerten (echt vs.
Verdacht). Kurz-Aliase: `haywood` (Default), `sdxl`, `ateeqq`, `umm`. Jede
beliebige HuggingFace-`image-classification`-Modell-ID ist auch direkt nutzbar.
Eigenen Vergleich fahren:
`.venv/bin/python tests/eval_model.py <id> ./echt ./verdacht`.

## Validierung

Gemessen an zwei realen Bildsätzen — bekannt echte Referenzfotos (Innenraum-/
Immobilienaufnahmen) vs. einem Satz Verdachtsbilder:

| Bildsatz | Ergebnis |
|---|---|
| Verdacht (7 Bilder) | **7/7 🔴** (Score 0.85–0.86) |
| echt (8 Bilder) | **7/8 🟢** (Score ~0.22), 1 False Positive |

Default-Modell `haywoodsloan/ai-image-detector-deploy` wurde empirisch gewählt
(beste Trennung, niedrigste FP-Rate: echt mean P(KI) 0.13 vs. KI 0.99). Eigene
Validierung mit deinen Ordnern:
`.venv/bin/python tests/eval_model.py haywood ./echt ./verdacht`.

## Grenzen (ehrlich)

- **False Positives** bei Fotos, die selbst synthetische Inhalte zeigen —
  abstrakte Kunst/Drucke an der Wand, Bildschirme, starke Stilisierung. (Der eine
  FP im Test war ein echtes Flurfoto mit abstrakten Wandbildern.)
- **False Negatives** bei neuen Generatoren (GPT-Image, Flux, Midjourney v6),
  nach Re-Encode, Downscaling oder Screenshot.
- Detektoren veralten schnell — das Modell ist deshalb über `--model` tauschbar.

## Modell tauschen

```bash
./aicheck.sh bild.png --model Organika/sdxl-detector
```

Eval-Helfer zum Vergleichen gegen die lokalen Datensätze:

```bash
.venv/bin/python tests/eval_model.py <hf-modell-id>
```

Die Label-Zuordnung (welche Klasse = „KI") erkennt gängige Namensschemata
automatisch (`fake/real`, `artificial/human`, …). Bei exotischen Labeln
(`LABEL_0/1`) liefert das Signal `n/a` statt zu raten.

## Online-Detektor: Sightengine (optional)

Als Zweitmeinung zum lokalen Modell. Standard-Provider ist
[Sightengine](https://sightengine.com):

1. Keys ins **`.env`** eintragen — **nicht** in `.env.example` (die wird ins Repo
   committet, dein Secret würde leaken!):
   ```bash
   cp .env.example .env      # .env mit deinen Keys füllen
   export $(grep -v '^#' .env | xargs)
   ```
2. Mit `--online` aufrufen:
   ```bash
   aicheck bild.png --online
   ```
   Im Report erscheint `online` als eigene Zeile **neben** `ml_local` — so
   stellst du lokal vs. Sightengine direkt gegenüber.

(`requests` kommt seit dem Setup automatisch mit.)

⚠️ `--online` **lädt das Bild zu Sightengine hoch** (US-Dienst); ohne das Flag
verlässt kein Bild den Rechner. Bei Kundenbildern DSGVO/AVV prüfen. Weitere
Provider lassen sich in `_call_provider()` (`aicheck/signals/online.py`) ergänzen.

### Kosten & Cache

Sightengines `genai`-Modell kostet **5 Operations pro Bild** (Free-Tier:
2.000 Ops/Monat ≈ 400 Bilder, danach ~1 Cent/Bild). Um nicht doppelt zu zahlen,
cached aicheck Online-Ergebnisse **lokal** in `~/.aicheck/online-cache.json`:

- Schlüssel ist der **SHA-256 des Bildinhalts** — greift auch nach Umbenennen;
  ein geändertes Bild bekommt automatisch einen neuen Call.
- Treffer kosten **0 Ops** (und brauchen nicht mal API-Keys).
- `--refresh` erzwingt einen Neu-Call (überschreibt den Eintrag).
- `--clear-cache` leert den Cache.

Nur erfolgreiche Ergebnisse werden gecacht — ein API-Fehler bleibt nicht hängen.

## Tests

Im aktivierten venv (jede Plattform):

```bash
python -m unittest tests.test_aicheck -v
```

Die optionalen Signal-Smoketests laufen nur, wenn du eigene Bildordner angibst:
`AICHECK_TEST_AI_DIR=… AICHECK_TEST_REAL_DIR=… python -m unittest …` (sonst werden
sie übersprungen).

## Aufbau

```
aicheck/
  pyproject.toml      Packaging + `aicheck`-Befehl + Dependencies
  setup.sh            Komfort-Setup (macOS/Linux/WSL)
  aicheck/
    cli.py            CLI + Orchestrierung
    scoring.py        Aggregation -> Ampel
    signals/
      base.py         SignalResult + Protokoll
      c2pa.py         C2PA-Manifest (hartes Signal)
      metadata.py     exiftool-Heuristik
      forensics.py    FFT-Hochfrequenz-Heuristik
      ml_local.py     HuggingFace-Detektor (Hauptsignal)
      online.py       Adapter-Stub
  tests/
  SPEC.md             Design-Dokument
```
