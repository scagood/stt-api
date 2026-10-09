#!/bin/bash
D=$(cd "$(dirname "$0")" && pwd)
S=$D/../wf65/.github/scripts/pull-request-section.sh
export FAKE_BODY_FILE=$(mktemp)
printf 'Intro text' > "$FAKE_BODY_FILE"
for i in 1 2 3 4; do
  sec=$(mktemp); printf 'run %s\n' "$i" > "$sec"
  PATH="$D/bin:$PATH" bash "$S" o/r 1 prerelease "$sec" 2>/dev/null
  printf 'after run %s: last 26 bytes: ' "$i"; tail -c 26 "$FAKE_BODY_FILE" | od -An -c | tr -s ' ' | tr '\n' ' '; echo
done
echo "--- removal:"; PATH="$D/bin:$PATH" bash "$S" o/r 1 prerelease 2>/dev/null; cat -A "$FAKE_BODY_FILE"; echo "<<EOF"
