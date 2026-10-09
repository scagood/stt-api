from parakeet_service import retime
w = lambda t, a, b: {"word": t, "start": a, "end": b}
words = [w("Then", 10.0, 11.8), w("said", 13.0, 13.4), w("him.", 14.0, 14.4), w("Next", 18.5, 19.0)]
print("crowded", retime._crowded((12.0, 18.0), words, [x["start"] for x in words], 0.0, 30.0))
print([(x["word"], round(x["start"], 2), round(x["end"], 2)) for x in retime.retime(words, [(12.0, 18.0)], 0.0, 30.0)])
# legit: before-then-after still moves
words2 = [w("Then", 10.0, 11.8), w("him.", 13.0, 13.4), w("said", 14.0, 14.4), w("Next", 18.5, 19.0)]
print([(x["word"], round(x["start"], 2), round(x["end"], 2)) for x in retime.retime(words2, [(12.0, 18.0)], 0.0, 30.0)])
