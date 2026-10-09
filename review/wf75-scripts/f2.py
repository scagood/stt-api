import sys
sys.path.insert(0, sys.argv[1])
from parakeet_service import retime as r
words = [{"word": "Then", "start": 10.0, "end": 11.8}, {"word": "said", "start": 13.0, "end": 13.4}, {"word": "him.", "start": 14.0, "end": 14.4}, {"word": "Next", "start": 18.5, "end": 19.0}]
out = r.retime(words, [(12.0, 18.0)], 0.0, 30.0)
for w in out: print(w["word"], round(w["start"], 3), round(w["end"], 3))
if hasattr(r, "_crowded"):
    print("crowded:", r._crowded((12.0, 18.0), words, [w["start"] for w in words], 0.0, 30.0))
    # suggested fix: crowded also when a 'before' word follows an 'after' word
    def crowded2(gap, words, starts, low, high):
        import bisect
        start, end = gap
        if start <= low and end >= high:
            return True
        before = after = 0
        for index in range(bisect.bisect_left(starts, start), len(words)):
            word = words[index]
            if word["start"] >= end:
                break
            if word["end"] <= end:
                side = r._before(word, gap, low, high)
                if side and after:
                    return True
                before, after = before + side, after + (not side)
        return before > 1 or after > 1
    r._crowded = crowded2
    print("with fix:")
    for w in r.retime(words, [(12.0, 18.0)], 0.0, 30.0): print(w["word"], round(w["start"], 3), round(w["end"], 3))
