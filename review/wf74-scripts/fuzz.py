import sys, random; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/wf74-scripts")
from load import *
CONFIGS = {
  "v2 ctx5": dict(target=25.0, mx=30.0, ctx=5.0),
  "v2 ctx0": dict(target=25.0, mx=30.0, ctx=0.0),
  "v3 ctx5": dict(target=60.0, mx=75.0, ctx=5.0),
  "whisper": dict(target=25.0, mx=30.0, ctx=0.0),
}
def layout(rng):
    t = rng.uniform(0, 4)
    segs = []
    for _ in range(rng.randint(1, 8)):
        L = rng.choice([rng.uniform(0.3, 6), rng.uniform(5, 25), rng.uniform(15, 50)])
        segs.append((round(t, 3), round(t + L, 3)))
        t += L + rng.choice([rng.uniform(0.05, 1.0), rng.uniform(0.5, 2.9), rng.uniform(3, 8)])
    total = round(segs[-1][1] + rng.choice([0, rng.uniform(0, 1.5), rng.uniform(1, 6)]), 3)
    return segs, total
if __name__ == "__main__":
  rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
  N = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
  for cname, cfg in CONFIGS.items():
      stats = dict(more74=0, fewer74=0, fixworse=0, fixbetter=0, chunked=0, slivers74=0, slivers_fix=0, pieces_fix_more=0)
      ex = None; exw = None
      for i in range(N):
          segs, total = layout(rng)
          if total <= cfg["mx"]:
              continue
          stats["chunked"] += 1
          res = {}
          for k, m in MODS.items():
              plan, sg, t = run(m, segs, total, **cfg)
              res[k] = (in_speech_cuts(plan.ranges, sg), plan.ranges, sg)
          a, b, c = res["55"][0], res["74"][0], res["fix"][0]
          if b > a:
              stats["more74"] += 1
              if ex is None: ex = (segs, total, show(res["55"][1]), show(res["74"][1]), show(res["fix"][1]))
          if b < a: stats["fewer74"] += 1
          if c > b:
              stats["fixworse"] += 1
              if exw is None: exw = (segs, total, show(res["74"][1]), show(res["fix"][1]))
          if c < b: stats["fixbetter"] += 1
          if len(res["fix"][1]) > len(res["74"][1]): stats["pieces_fix_more"] += 1
          # sliver = range shorter than 2 s that meets a neighbour
          for key, sk in (("74", "slivers74"), ("fix", "slivers_fix")):
              r = res[key][1]
              if any(e - s < 2 * SR and ((j and r[j-1][1] == s) or (j + 1 < len(r) and r[j+1][0] == e)) for j, (s, e) in enumerate(r)):
                  stats[sk] += 1
      print(cname, stats)
      if ex: print("   example 74 worse than 55:", ex)
      if exw: print("   example fix worse than 74:", exw)
