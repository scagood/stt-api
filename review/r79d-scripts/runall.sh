#!/bin/bash
cd /tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/r79d-scripts
P=/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/venv/bin/python
N=${N:-1500}
for cfg in "c1 101 jit=0.1 mis=0.07 madeup=0 persist=0.35 fresh=0.1" "c2 202 jit=0.2 mis=0.10 madeup=0 persist=0.8 fresh=0.2" "c3 303 jit=0.1 mis=0.05 madeup=0 persist=0.0 fresh=0.0" "c4 404 jit=0.1 mis=0.07 madeup=0.3 persist=0.35 fresh=0.1"; do
  set -- $cfg; name=$1; seed=$2; shift 2; opts="$*"
  for mode in one two three batch; do
    for w in $WTS; do
      echo "$P fz.py /tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/r79d-$w $mode $seed $N out/$name-$w-$mode.jsonl $opts"
    done
  done
done | xargs -P ${J:-4} -I{} bash -c "{}"
