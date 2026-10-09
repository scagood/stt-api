import json, glob, os, math
D = os.path.dirname(os.path.abspath(__file__))
rows = {}
for f in sorted(glob.glob(f"{D}/out_*_4242.json")):
    rows.update(json.load(open(f)))
tot = {}
print(f"{'bucket/scale/config':40s} {'L':>4s} {'pieces':>11s} {'forced':>11s} {'main inword':>14s} {'head inword':>14s} worse/better  <2s  silent lost bad q  drop(inword)  ctx-edge in word main->head (worse/better)")
for k, C in rows.items():
    g = lambda x: C.get(x, 0)
    print(f"{k:40s} {g('layouts'):4d} {g('m_n'):5d}->{g('h_n'):5d} {g('m_forced'):5d}->{g('h_forced'):5d} {g('m_inword'):5d} ({g('m_inword')/max(1,g('m_forced')):5.1%}) {g('h_inword'):5d} ({g('h_inword')/max(1,g('h_forced')):5.1%}) {g('worse_inword'):3d}/{g('better_inword'):3d}"
          f"  {g('m_short')}->{g('h_short')} {g('m_silent')}->{g('h_silent')} {g('m_lost')}->{g('h_lost')} {g('m_bad')}->{g('h_bad')} {g('quarter_viol')} {g('drop_layouts'):3d}({g('drop_in_word')},{g('drop_in_speech')})"
          f"  {g('m_edge_word')}/{g('m_edges')} ({g('m_edge_word')/max(1,g('m_edges')):.1%}) -> {g('h_edge_word')}/{g('h_edges')} ({g('h_edge_word')/max(1,g('h_edges')):.1%}) ({g('edge_word_worse')}/{g('edge_word_better')}) speech {g('m_edge_speech')}->{g('h_edge_speech')}")
    cfg = k.split(" ", 2)[2]
    for key in ("m_edge_word", "h_edge_word", "m_edges", "h_edges", "edge_word_worse", "edge_word_better", "m_edge_speech", "h_edge_speech", "m_inword", "h_inword", "m_forced", "h_forced", "worse_inword", "better_inword", "layouts", "worse_n", "worse_short", "worse_silent", "worse_lost", "h_bad", "quarter_viol", "drop_in_word", "drop_in_speech"):
        tot.setdefault(cfg, {}).setdefault(key, 0); tot[cfg][key] += g(key)
print()
for cfg, t in tot.items():
    w, b = t["edge_word_worse"], t["edge_word_better"]
    n = w + b; z = (w - b) / math.sqrt(n) if n else 0
    print(f"{cfg}: layouts {t['layouts']}, inword {t['m_inword']}/{t['m_forced']} ({t['m_inword']/max(1,t['m_forced']):.1%}) -> {t['h_inword']}/{t['h_forced']} ({t['h_inword']/max(1,t['h_forced']):.1%}), layouts worse/better {t['worse_inword']}/{t['better_inword']}; "
          f"worse n/short/silent/lost {t['worse_n']}/{t['worse_short']}/{t['worse_silent']}/{t['worse_lost']} bad {t['h_bad']} q {t['quarter_viol']} drop in word/speech {t['drop_in_word']}/{t['drop_in_speech']}; "
          f"ctx edges in word {t['m_edge_word']}/{t['m_edges']} ({t['m_edge_word']/max(1,t['m_edges']):.2%}) -> {t['h_edge_word']}/{t['h_edges']} ({t['h_edge_word']/max(1,t['h_edges']):.2%}); in speech {t['m_edge_speech']} -> {t['h_edge_speech']}; layouts more/fewer edge-in-word {w}/{b} (sign z={z:+.2f})")
