import json
from collections import Counter

data = json.load(open('items.json', 'r'))
diff = [abs(x[0] - x[1]) for x in data]

print(Counter(diff))

interesting = list(filter(lambda x: abs(x[0] - x[1]) > 7, data))
json.dump(interesting, open('interesting.json', 'w'), indent=4)