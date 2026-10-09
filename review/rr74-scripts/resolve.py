import re, sys
p = sys.argv[1]
src = open(p).read()
# take PR's side of the conflict hunks, then adapt
def take(src, side):
    out = []; mode = None
    for line in src.splitlines(keepends=True):
        if line.startswith("<<<<<<<"): mode = "ours"; continue
        if line.startswith("=======") and mode: mode = "theirs"; continue
        if line.startswith(">>>>>>>") and mode: mode = None; continue
        if mode is None or mode == side: out.append(line)
    return "".join(out)
src = take(src, "theirs")
print(src.count("_split_oversized("), "calls")
open(p, "w").write(src)
