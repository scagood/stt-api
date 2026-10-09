#!/bin/bash
# usage: run.sh <script> <bodyfile-in> [sectionfile]  -> prints resulting body (visible) and edit count
S=/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/rr65-scripts
script=$1; shift
export STUB_BODY=$(mktemp); cp "$1" "$STUB_BODY"; shift
rm -f "$STUB_BODY.edits"
PATH=$S/bin:$PATH bash "$script" o/r 65 prerelease "$@" >/dev/null
cat -A "$STUB_BODY"; echo "edits=$(cat "$STUB_BODY.edits" 2>/dev/null | wc -l)"
