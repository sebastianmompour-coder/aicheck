#!/usr/bin/env bash
# Bequemes Setup für macOS/Linux/WSL. Windows: siehe README ("Setup").
# Legt ein venv an, installiert aicheck + Dependencies, prüft externe Tools.
set -euo pipefail
cd "$(dirname "$0")"

PY=python3
for c in python3.14 python3.13 python3.12 python3.11; do
  command -v "$c" >/dev/null 2>&1 && { PY="$c"; break; }
done

echo "==> venv anlegen ($("$PY" --version))"
"$PY" -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip
echo "==> aicheck + Dependencies installieren (torch ist gross, dauert)"
.venv/bin/python -m pip install -e .

echo "==> externe Tools prüfen (optional; fehlend -> Signal liefert n/a)"
command -v exiftool >/dev/null 2>&1 || echo "   exiftool fehlt  -> macOS: brew install exiftool | Linux: apt install libimage-exiftool-perl"
command -v c2patool >/dev/null 2>&1 || echo "   c2patool fehlt  -> macOS: brew install c2patool | sonst: cargo install c2patool"

echo "==> fertig. Nutzung:"
echo "   .venv/bin/aicheck /pfad/zum/ordner      (oder: source .venv/bin/activate && aicheck ...)"
