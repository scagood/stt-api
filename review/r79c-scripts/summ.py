import subprocess, sys, re, os
P = "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/venv/bin/python"
keys = ["cases", "lost_words", "dup_words", "HEAD_EXTRA_COPIES_vs_main", "main_missing", "head_missing", "main_invented", "head_invented",
        "main_decodes", "head_decodes", "main_sec", "head_sec", "head_vad_calls"]
bad = re.compile(r"^(LOST|DUP|REORDER|head_crash|main_crash|head_(text|unsorted|word|verbose|srt|path)|head_more_truth)")
cfgs = sys.argv[1:] or ["d1", "d2", "d3", "d4", "d5"]
print("cfg mode " + " ".join(keys) + " | problems")
for c in cfgs:
    for m in ["one", "two", "three", "batch"]:
        a, b = f"out/{c}-main-{m}.jsonl", f"out/{c}-head-{m}.jsonl"
        if not (os.path.exists(a) and os.path.exists(b)):
            continue
        out = subprocess.run([P, "cmp.py", a, b, "3"], capture_output=True, text=True).stdout
        d = {}
        probs = []
        for line in out.splitlines():
            if ": " in line and not line.startswith(("---", "    ")):
                k, v = line.split(": ", 1)
                d[k] = v
                if bad.match(k):
                    probs.append(f"{k}={v}")
        print(c, m, " ".join(d.get(k, "0") for k in keys), "|", " ".join(probs))
