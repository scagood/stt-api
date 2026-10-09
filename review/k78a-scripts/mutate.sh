#!/bin/bash
S=/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad
W=$S/k78a-mut; F=$W/parakeet_service/chunker.py
cd $W
run() { $S/venv/bin/python -m pytest -q -p no:cacheprovider -x tests/test_chunker.py tests/test_retime.py >/dev/null 2>&1 && echo pass || echo FAIL; }
git checkout -q -- parakeet_service/chunker.py
echo "baseline: $(run)"
out=""
for v in 18 19 20 21 22 23 24 25 26 27 28 29 30; do
  git checkout -q -- parakeet_service/chunker.py
  sed -i "s/^_HEARD_ENOUGH = 25 /_HEARD_ENOUGH = $v /" $F
  grep -q "^_HEARD_ENOUGH = $v " $F || { echo "sed failed"; exit 1; }
  out="$out $v:$(run)"
done
echo "_HEARD_ENOUGH$out"
out=""
for v in 10 12 14 15 16 17 18 19 20 21 22 23 24 25 26 30; do
  git checkout -q -- parakeet_service/chunker.py
  sed -i "s/^_SOUND_DIP = 20 /_SOUND_DIP = $v /" $F
  grep -q "^_SOUND_DIP = $v " $F || { echo "sed failed"; exit 1; }
  out="$out $v:$(run)"
done
echo "_SOUND_DIP$out"
out=""
for f in 0.5 0.7 0.8 0.85 0.88 0.89 0.9 0.91 0.95 1.0 1.2 1.5 2.0 3.0; do
  git checkout -q -- parakeet_service/chunker.py
  sed -i "s/near\[max(0, a - relisten): b + relisten\] = True/near[max(0, a - int(relisten * $f)): b + int(relisten * $f)] = True/" $F
  grep -q "relisten \* $f" $F || { echo "sed failed"; exit 1; }
  out="$out $f:$(run)"
done
echo "near x$out"
git checkout -q -- parakeet_service/chunker.py
git status --short
