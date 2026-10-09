#!/bin/bash
cd /tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/r79d-scripts
P=/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/venv/bin/python
N=${N:-1000}
for cfg in "k1 11" "k2 22 far=0 shift=0.3" "k3 33 far=0 shift=0 endearly=0" "k4 44 func=0.8 rep=0.3 dense=1 far=0"; do
  set -- $cfg; name=$1; seed=$2; shift 2; opts="$*"
  for mode in one two three batch; do
    for w in $WTS; do
      echo "$P cutfz.py /tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/r79d-$w $mode $seed $N cout/$name-$w-$mode.jsonl $opts"
    done
  done
done | xargs -P ${J:-2} -I{} bash -c "{}"
