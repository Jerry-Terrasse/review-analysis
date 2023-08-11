import json
import sys
from random import shuffle

fname = sys.argv[1]
with open(fname) as f:
    data: list[tuple[float, str]] = json.load(f)


results: dict[float, list[str]] = {r/2: [] for r in range(1, 11)}
for rating, quote in data:
    results[rating].append((rating, quote))

train_ratio = 0.8
train_data = []
test_data = []

for rating, quotes in results.items():
    shuffle(quotes)
    cut = round(len(quotes) * train_ratio)
    print(f"Rating {rating}: using {cut} for training, {len(quotes) - cut} for testing")
    train_data.extend(quotes[:cut])
    test_data.extend(quotes[cut:])

json.dump(train_data, open(fname[:-5] + '_train.json', 'w'), indent=4)
json.dump(test_data, open(fname[:-5] + '_test.json', 'w'), indent=4)