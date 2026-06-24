#!/usr/bin/env bash
# Wrapper: ruft aicheck im projekteigenen venv auf.
DIR="$(cd "$(dirname "$0")" && pwd)"
if [ ! -x "$DIR/.venv/bin/python" ]; then
    echo "venv fehlt. Erst ./setup.sh ausführen." >&2
    exit 1
fi
exec "$DIR/.venv/bin/python" -m aicheck "$@"
