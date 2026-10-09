#!/bin/bash
S=/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad
cd $S/r79c-scripts
P=$S/venv/bin/python
for cfg in "d1 7101 jit=0.1 mis=0.07 persist=0.35 fresh=0.1" "d2 7202 jit=0.2 mis=0.10 persist=0.8 fresh=0.2" "d3 7303 jit=0.1 mis=0.05 persist=0.0 fresh=0.0" "d4 7404 jit=0.1 mis=0.07 madeup=0.3 persist=0.35 fresh=0.1" "d5 7505 jit=0.2 mis=0.10 madeup=0.3 mmid=0.5 mstart=0.1 persist=0.8 fresh=0.2"; do
  set -- $cfg; name=$1; seed=$2; shift 2; opts="$*"
  for mode in one two three batch; do
    for w in main head; do
      echo "$P fz.py $S/r79c-$w $mode $seed 2000 out/$name-$w-$mode.jsonl $opts"
    done
  done
done | xargs -P 4 -I{} bash -c "{}"
