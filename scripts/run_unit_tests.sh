#!/usr/bin/env bash
# Run unit tests only. Used by pre-commit hook to block commits on failures.
set -euo pipefail
exec uv run pytest -m unit -x -q
