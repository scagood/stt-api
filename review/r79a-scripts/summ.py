import subprocess, sys, re
P = sys.executable
keys = ["cases","files","truth_words","main_missing","head_missing","lost_words","dup_words","REORDER_of_main_words","HEAD_EXTRA_COPIES_vs_main","main_invented","head_invented","head_crash","main_crash","main_decodes","head_decodes","main_sec","head_sec","audio_sec","head_vad_calls","paths_run"]
print("cfg mode | " + " ".join(keys) + " | head problems")
for c in ("c1","c2","c3","c4"):
    for m in ("one","two","three","batch"):
        out = subprocess.run([P,"cmp.py",f"out/{c}-main-{m}.jsonl",f"out/{c}-head-{m}.jsonl","0"],capture_output=True,text=True).stdout
        d = dict(re.findall(r"^([\w.\- /<>!=]+): ([\d.]+)$", out, re.M))
        probs = {k: v for k, v in d.items() if (k.startswith("head_") and k not in keys and k not in ("head_more_invented","head_extra_copy_case","head_truth_inversions")) or k.startswith("head_path")}
        print(f"{c} {m} | " + " ".join(f"{k.split('_')[0][:1]}{d.get(k,'0')}" if False else str(d.get(k,'0')) for k in keys) + f" | {probs}")
