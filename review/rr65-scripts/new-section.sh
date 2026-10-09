#!/bin/bash
set -euo pipefail

# Replace -- or remove -- a managed section of a pull request body. The section
# is delimited by a named pair of HTML comments, so repeated runs rewrite it in
# place rather than stacking copies of it, and so separate groups can sit in the
# same body without disturbing each other.
#
#   pull-request-section.sh <owner/repo> <pr-number> <comment-group> [section-file]
#
# Called without a section file the section is removed.

repository="${1:?an owner/repo is required}"
pullNumber="${2:?a pull request number is required}"
commentGroup="${3:?a comment group is required}"
sectionFile="${4:-}"

startMarker="<!-- ${commentGroup}:start -->"
endMarker="<!-- ${commentGroup}:end -->"

workDir="$(mktemp -d)"
trap 'rm -rf "${workDir}"' EXIT

bodyFile="${workDir}/body.md"
nextBodyFile="${workDir}/next-body.md"

# GitHub stores bodies with CRLF line endings, so normalise them before
# looking for the markers. `--jq` prints a newline after the body and the
# command substitution drops every trailing one, so the file ends in exactly
# one newline: a body this script wrote reads back unchanged, and rewriting
# the section doesn't add a blank line each time.
body="$(
  gh pr view "${pullNumber}" --repo "${repository}" --json body --jq '.body // ""' \
    | tr -d '\r'
)"
if [ -n "${body}" ]; then
  printf '%s\n' "${body}" > "${bodyFile}"
else
  : > "${bodyFile}"
fi

replacementFile=""
if [ -n "${sectionFile}" ]; then
  replacementFile="${workDir}/replacement.md"
  {
    printf '%s\n' "${startMarker}"
    cat "${sectionFile}"
    printf '%s\n' "${endMarker}"
  } > "${replacementFile}"
fi

# How the marker lines pair up, ignoring spaces around them. Only exactly one
# start above exactly one end is a section to rewrite.
markers="$(
  awk \
    -v startMarker="${startMarker}" \
    -v endMarker="${endMarker}" '
      { trimmed = $0; gsub(/^[ \t]+|[ \t]+$/, "", trimmed) }
      trimmed == startMarker { starts++; startAt = NR }
      trimmed == endMarker { ends++; endAt = NR }
      END {
        if (starts == 1 && ends == 1 && startAt < endAt) print "paired"
        else if (starts + ends > 0) print "stray"
        else print "none"
      }
    ' "${bodyFile}"
)"

sourceFile="${bodyFile}"
if [ "${markers}" = stray ]; then
  # A marker was deleted, doubled or moved by hand. Pairing up what is left
  # could swallow text that isn't the section, so only the marker lines go,
  # and the body is treated as never having carried the section.
  sourceFile="${workDir}/stripped.md"
  awk \
    -v startMarker="${startMarker}" \
    -v endMarker="${endMarker}" '
      { trimmed = $0; gsub(/^[ \t]+|[ \t]+$/, "", trimmed) }
      trimmed != startMarker && trimmed != endMarker { print }
    ' "${bodyFile}" > "${sourceFile}"
fi

if [ "${markers}" = paired ]; then
  # Rewrite the section where it stands -- moving it back to the bottom would
  # step over any other group's section on the way.
  awk \
    -v startMarker="${startMarker}" \
    -v endMarker="${endMarker}" \
    -v replacementFile="${replacementFile}" '
      { trimmed = $0; gsub(/^[ \t]+|[ \t]+$/, "", trimmed) }
      trimmed == startMarker {
        skip = 1
        while (replacementFile != "" && (getline line < replacementFile) > 0) {
          print line
        }
        next
      }
      skip && trimmed == endMarker { skip = 0; next }
      !skip { print }
    ' "${bodyFile}" > "${nextBodyFile}"
elif [ -n "${replacementFile}" ]; then
  # A section the body doesn't carry goes at the bottom. The command
  # substitution drops trailing newlines, so it sits flush against whatever
  # the body happened to end with.
  body="$(cat "${sourceFile}")"
  {
    if [ -n "${body}" ]; then
      printf '%s\n\n' "${body}"
    fi
    cat "${replacementFile}"
  } > "${nextBodyFile}"
else
  cp "${sourceFile}" "${nextBodyFile}"
fi

if cmp -s "${bodyFile}" "${nextBodyFile}"; then
  echo "The pull request body is already up to date"
  exit 0
fi

gh pr edit "${pullNumber}" --repo "${repository}" --body-file "${nextBodyFile}"
