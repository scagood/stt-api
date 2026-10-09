#!/bin/bash
# usage: run.sh <initial-body-printf-format> [section text]
S=/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/wf65/.github/scripts/pull-request-section.sh
export FAKE_BODY_FILE=$(mktemp)
printf "$1" > "$FAKE_BODY_FILE"
if [ -n "${2-}" ]; then sec=$(mktemp); printf '%s\n' "$2" > "$sec"; else sec=""; fi
PATH="$(dirname "$0")/bin:$PATH" bash "$S" o/r 1 prerelease $sec
echo "--- stored body (od -c tail):"; cat -A "$FAKE_BODY_FILE"; echo "<<EOF"
