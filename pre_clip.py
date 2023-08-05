import json
import sys

fname = sys.argv[1]
with open(fname) as f:
    data: list[tuple[float, str]] = json.load(f)

result = []
for rating, quote in data:
    quote = quote.split()
    for i in range(len(quote) // 100 + 1):
        piece = quote[i*100: (i+1)*100]
        if len(piece) < 50:
            break
        result.append((rating, ' '.join(piece)))

json.dump(result, open(fname[:-5] + '_clip.json', 'w'), indent=4)