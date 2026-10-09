import sys, re
src = open(sys.argv[1]).read()
variant = sys.argv[2]
blocks = re.findall(r'<<<<<<< c032ee4\n(.*?)=======\n(.*?)>>>>>>> 04fc7a9\n', src, re.S)
assert len(blocks) == 2
ours_cut, theirs_cut = blocks[0]
ours_last, theirs_last = blocks[1]
if variant == 'A':
    cut, last = ours_cut, ours_last
else:
    cut = ("        # B: slide-later term kept, capped at _furthest_end\n"
           "        cut = min(max((current_end + start) // 2, min(end - own_maximum, start)), _furthest_end(current_start, current_end, own_maximum))\n")
    last = ours_last
out = src.replace(f'<<<<<<< c032ee4\n{ours_cut}=======\n{theirs_cut}>>>>>>> 04fc7a9\n', cut, 1)
out = out.replace(f'<<<<<<< c032ee4\n{ours_last}=======\n{theirs_last}>>>>>>> 04fc7a9\n', last, 1)
if variant == 'C':
    old = ("    if current_end - segments[0][0] <= own_maximum:\n"
           "        current_start = max(current_start, current_end - own_maximum)\n")
    assert old in out
    out = out.replace(old, "    current_start = max(current_start, current_end - -(-(current_end - segments[0][0]) // own_maximum) * own_maximum)\n")
open(sys.argv[3], 'w').write(out)
