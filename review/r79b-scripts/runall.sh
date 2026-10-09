#!/bin/bash
S=/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad
cd $S/r79b-scripts
run() { tag=$1; shift; for w in main head merged; do echo "WT=$S/r79b-$w $S/venv/bin/python fuzz2.py $* > out/$tag-$w.jsonl"; done; }
{
run A 11 1500 mix
run B2 12 800 2 cut=1
run B3 13 800 3 cut=1
run C 14 1000 mix split=0.03
run D 15 1000 mix frag=1
run E 16 1000 mix del=0.05
run F 17 1000 mix mis=0.1 jit=0.6
run G 18 1000 1 stretchskip=0.5 split=0.02 del=0.03 frag=1
} | xargs -P 4 -I{} bash -c '{}'
