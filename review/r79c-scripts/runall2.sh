#!/bin/bash
S=/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad
cd $S/r79c-scripts
run() { tag=$1; shift; for w in main head; do echo "WT=$S/r79c-$w $S/venv/bin/python fuzz2.py $* > out2/$tag-$w.jsonl"; done; }
{
run A 811 1500 mix
run B2 812 1000 2 cut=1
run B3 813 1000 3 cut=1
run C 814 1200 mix split=0.05
run D 815 1200 mix frag=1
run E 816 1200 mix del=0.05
run F 817 1200 mix mis=0.1 jit=0.6
run G 818 1200 1 stretchskip=0.5 split=0.03 del=0.03 frag=1
run H 819 1000 mix rangeskip=0.0 split=0.03
} | xargs -P 4 -I{} bash -c '{}'
