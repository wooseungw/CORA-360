#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python -m pip install -r "$ROOT/requirements-repro.txt"
python - <<'PY'
import nltk

for resource in ("wordnet", "punkt", "punkt_tab"):
    nltk.download(resource, quiet=True)
print("NLTK metric resources installed")
PY

if ! command -v java >/dev/null 2>&1; then
  echo "SPICE requires Java. On Ubuntu/Debian: sudo apt-get install default-jre-headless" >&2
  exit 1
fi

python "$ROOT/scripts/baseline_eval.py" --help
echo "Caption metric dependencies are ready."
