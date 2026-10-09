#!/bin/bash
S=/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad
W=$S/v78-mut; F=$W/parakeet_service/chunker.py
cd $W
run() { $S/venv/bin/python -m pytest -q -p no:cacheprovider -x tests/test_chunker.py tests/test_retime.py >/dev/null 2>&1 && echo pass || echo FAIL; }
run1() { $S/venv/bin/python -m pytest -q -p no:cacheprovider -x "$@" >/dev/null 2>&1 && echo pass || echo FAIL; }
git checkout -q -- parakeet_service/chunker.py
echo "baseline: $(run)"
out=""
for v in 22 23 24 25 26 27 28 29; do
  git checkout -q -- parakeet_service/chunker.py
  sed -i "s/^_HEARD_ENOUGH = 25 /_HEARD_ENOUGH = $v /" $F; grep -q "^_HEARD_ENOUGH = $v " $F || { echo "sed failed"; exit 1; }
  out="$out $v:$(run)"
done
echo "_HEARD_ENOUGH (both uses)$out"
out=""
for v in 14 15 16 17 18 19 20 21 22 23 24; do
  git checkout -q -- parakeet_service/chunker.py
  sed -i "s/^_SOUND_DIP = 20 /_SOUND_DIP = $v /" $F; grep -q "^_SOUND_DIP = $v " $F || { echo "sed failed"; exit 1; }
  out="$out $v:$(run)"
done
echo "_SOUND_DIP$out"
out=""
for v in 0 5 10 15 16 17 18 19 20 25 28 29 30 31 35 40 60 100; do
  git checkout -q -- parakeet_service/chunker.py
  sed -i "s/reach\[max(0, a - relisten - _HEARD_ENOUGH): b + relisten + _HEARD_ENOUGH\] = True/reach[max(0, a - relisten - $v): b + relisten + $v] = True/" $F; grep -q "relisten + $v\]" $F || { echo "sed failed"; exit 1; }
  out="$out $v:$(run)"
done
echo "reach pad$out"
out=""; out1=""
for f in 0.8 0.83 0.84 0.85 0.9 0.95 0.97 0.98 0.99 1.0 1.2; do
  git checkout -q -- parakeet_service/chunker.py
  sed -i "s/near\[max(0, a - relisten): b + relisten\] = True/near[max(0, a - int(relisten * $f)): b + int(relisten * $f)] = True/" $F; grep -q "relisten \* $f" $F || { echo "sed failed"; exit 1; }
  out="$out $f:$(run)"
  out1="$out1 $f:$(run1 'tests/test_chunker.py::test_volume_keeps_a_quiet_word_near_a_phrase_whole')"
done
echo "near x (reach unchanged), suite$out"
echo "near x, word tests alone$out1"
out=""
for f in 0.9 0.95 0.97 0.98 0.99 1.0; do
  git checkout -q -- parakeet_service/chunker.py
  sed -i "s/near\[max(0, a - relisten): b + relisten\] = True/near[max(0, a - int(relisten * $f)): b + int(relisten * $f)] = True/; s/reach\[max(0, a - relisten - _HEARD_ENOUGH): b + relisten + _HEARD_ENOUGH\] = True/reach[max(0, a - int(relisten * $f) - _HEARD_ENOUGH): b + int(relisten * $f) + _HEARD_ENOUGH] = True/" $F
  out="$out $f:$(run)"
done
echo "near and reach x$out"
git checkout -q -- parakeet_service/chunker.py
git status --short
