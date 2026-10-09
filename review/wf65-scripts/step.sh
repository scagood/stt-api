# The label can come off while the images build. That run drops the
# section, so this one must not write it back.
labels=$(gh pr view "$PR_NUMBER" --repo "$GITHUB_REPOSITORY" --json labels --jq '.labels[].name')
if [ "$ACTION" = unlabeled ] || ! grep -qx prerelease <<< "$labels"; then
  echo SECTION-SCRIPT "$GITHUB_REPOSITORY" "$PR_NUMBER" prerelease
  exit 0
fi

section="$RUNNER_TEMP/section.md"
# The backticks below are Markdown, not command substitution.
# shellcheck disable=SC2016
{
  echo '### Prerelease images'
  echo
  if [ "$CPU" = success ] || [ "$GPU" = success ]; then
    echo '```bash'
    if [ "$CPU" = success ]; then echo "docker pull $IMAGE-cpu"; fi
    if [ "$GPU" = success ]; then echo "docker pull $IMAGE-gpu"; fi
    echo '```'
    echo
  fi

  # Anything that failed goes after everything that worked. Its tag
  # may hold this commit's broken image, or an older one.
  if [ "$CPU" != success ]; then
    printf -- '- :x: `pr-%s-cpu` failed its build or smoke test, so do not pull it\n' "$PR_NUMBER"
  fi
  if [ "$GPU" != success ]; then
    printf -- '- :x: `pr-%s-gpu` failed to build, so do not pull it\n' "$PR_NUMBER"
  fi
  if [ "$CPU" != success ] || [ "$GPU" != success ]; then echo; fi

  echo 'For `linux/amd64`, rebuilt on each push while the `prerelease` label is on, and deleted when the pull request closes.'
  echo
  printf 'Built from %s &middot; [workflow run](%s)\n' "${HEAD_SHA:0:7}" "$RUN_URL"
} > "$section"

echo SECTION-SCRIPT \
  "$GITHUB_REPOSITORY" "$PR_NUMBER" prerelease "$section"
