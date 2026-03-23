#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/venv310}"
PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/bin/python3.10}"
REQ_FILE="$ROOT_DIR/requirements.txt"
STAMP_FILE="$VENV_DIR/.deps-installed"
FORCE_PIP_SYNC="${FORCE_PIP_SYNC:-0}"
OPEN_BROWSER="${OPEN_BROWSER:-1}"
BROWSER_URL="${BROWSER_URL:-http://127.0.0.1:5000/}"
BROWSER_CMD="${BROWSER_CMD:-}"

launch_browser() {
  local url="$1"
  if [[ "$OPEN_BROWSER" != "1" ]]; then
    return
  fi

  log "Opening browser at $url"

  if [[ -n "$BROWSER_CMD" ]]; then
    # Allow user-specified launcher with optional args
    # shellcheck disable=SC2206
    local cmd=( $BROWSER_CMD )
    cmd+=( "$url" )
    if "${cmd[@]}" >/dev/null 2>&1; then
      return
    fi
    log "Browser cmd '$BROWSER_CMD' failed; falling back"
  fi

  if command -v open >/dev/null 2>&1; then
    open "$url" >/dev/null 2>&1 || true
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$url" >/dev/null 2>&1 || true
  else
    log "Could not find a browser launcher (set BROWSER_CMD)"
  fi
}

log() {
  printf '[run_main] %s\n' "$1"
}

die() {
  printf '[run_main] ERROR: %s\n' "$1" >&2
  exit 1
}

APP_PID=""

cleanup() {
  if [[ -z "$APP_PID" ]]; then
    return
  fi
  if kill -0 "$APP_PID" >/dev/null 2>&1; then
    log "Stopping main.py"
    kill "$APP_PID"
  fi
}

if [[ ! -f "$REQ_FILE" ]]; then
  die "requirements.txt not found at $REQ_FILE"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  if command -v python3.10 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.10)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
    log "Falling back to $(basename "$PYTHON_BIN"); ensure it is Python 3.10 or 3.11"
  else
    die "Could not locate a usable python3.10 interpreter; set PYTHON_BIN to override"
  fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating virtual environment at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  NEED_PIP_SYNC=1
else
  NEED_PIP_SYNC=0
fi

if [[ "$FORCE_PIP_SYNC" == "1" ]]; then
  NEED_PIP_SYNC=1
elif [[ ! -f "$STAMP_FILE" || "$REQ_FILE" -nt "$STAMP_FILE" ]]; then
  NEED_PIP_SYNC=1
fi

VENV_PY="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

if [[ "$NEED_PIP_SYNC" == "1" ]]; then
  log "Installing Python dependencies"
  "$VENV_PY" -m pip install --upgrade pip setuptools wheel
  "$VENV_PIP" install -r "$REQ_FILE"
  touch "$STAMP_FILE"
fi

log "Launching main.py"
"$VENV_PY" "$ROOT_DIR/main.py" "$@" &
APP_PID=$!
trap cleanup INT TERM

launch_browser "$BROWSER_URL"

wait "$APP_PID"
EXIT_CODE=$?
trap - INT TERM
exit "$EXIT_CODE"
