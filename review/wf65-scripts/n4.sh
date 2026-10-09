#!/bin/bash
D=$(cd "$(dirname "$0")" && pwd)
export ACTION=synchronize CPU= GPU= HEAD_SHA=e67aa3f1bcbb IMAGE=ghcr.io/scagood/stt-api:pr-65 PR_NUMBER=65 RUN_URL=https://x/run GITHUB_REPOSITORY=scagood/stt-api RUNNER_TEMP=$D/rt
PATH="$D/bin2:$PATH" bash -e "$D/step.sh"; cat "$D/rt/section.md"
