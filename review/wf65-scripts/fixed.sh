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
# looking for the markers.
gh pr view "${pullNumber}" --repo "${repository}" --json body --jq '.body // ""' \
  | tr -d '\r' > "${bodyFile}"

replacementFile=""
if [ -n "${sectionFile}" ]; then
  replacementFile="${workDir}/replacement.md"
  {
    printf '%s\n' "${startMarker}"
    cat "${sectionFile}"
    printf '%s\n' "${endMarker}"
  } > "${replacementFile}"
fi

startLine=$(grep -nxF -- "${startMarker}" "${bodyFile}" | head -n1 | cut -d: -f1 || true); endLine=$(grep -nxF -- "${endMarker}" "${bodyFile}" | head -n1 | cut -d: -f1 || true)
if [ -n "$startLine" ] && [ -n "$endLine" ] && [ "$startLine" -lt "$endLine" ]; then
  # Rewrite the section where it stands -- moving it back to the bottom would
  # step over any other group's section on the way.
  awk \
    -v startMarker="${startMarker}" \
    -v endMarker="${endMarker}" \
    -v replacementFile="${replacementFile}" '
      $0 == startMarker {
        skip = 1
        while (replacementFile != "" && (getline line < replacementFile) > 0) {
          print line
        }
        next
      }
      skip && $0 == endMarker { skip = 0; next }
      !skip { print }
    ' "${bodyFile}" > "${nextBodyFile}"
elif [ -n "${replacementFile}" ]; then
  # A section the body has never carried goes at the bottom. The command
  # substitution drops trailing newlines, so it sits flush against whatever
  # the body happened to end with.
  body="$(cat "${bodyFile}")"
  {
    if [ -n "${body}" ]; then
      printf '%s\n\n' "${body}"
    fi
    cat "${replacementFile}"
  } > "${nextBodyFile}"
else
  cp "${bodyFile}" "${nextBodyFile}"
fi

if cmp -s "${bodyFile}" "${nextBodyFile}"; then
  echo "The pull request body is already up to date"
  exit 0
fi

gh pr edit "${pullNumber}" --repo "${repository}" --body-file "${nextBodyFile}"
