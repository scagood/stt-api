#!/bin/bash
S=$(dirname "$0")/fixed.sh
export FAKE_BODY_FILE=$(mktemp)
printf "$1" > "$FAKE_BODY_FILE"
for s in NEW1 NEW2; do sec=$(mktemp); printf '%s\n' "$s" > "$sec"; PATH="$(dirname "$0")/bin:$PATH" bash "$S" o/r 1 prerelease $sec; echo "--- after $s:"; cat -A "$FAKE_BODY_FILE"; echo "<<EOF"; done
