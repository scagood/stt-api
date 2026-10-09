#!/bin/bash
# usage: run.sh <script> <bodyfile-init-content-file> [section-file]
S=$(dirname "$0")
export PATH="$S/bin:$PATH"
export STUB_BODY=$(mktemp) STUB_EDITS=$(mktemp)
cp "$2" "$STUB_BODY"
"$1" o/r 65 prerelease ${3:+"$3"} >/dev/null
cat "$STUB_BODY"
