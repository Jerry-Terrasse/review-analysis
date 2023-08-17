import json
import sys
from tqdm import tqdm
from collections import Counter

from langdetect import detect

fname = sys.argv[1]
with open(fname, 'r') as f:
    data = json.load(f)

langs = []
filtered = []
for r, q in tqdm(data):
    try:    
        lang = detect(q)
    except Exception as e:
        print(f"Error: {e}, {q}")
        continue
    langs.append(lang)
    if lang == 'en':
        filtered.append((r, q))

counter = Counter(langs)
print(counter)

with open(sys.argv[2], 'w') as f:
    json.dump(filtered, f, indent=2)