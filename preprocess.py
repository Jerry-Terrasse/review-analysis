import re
import glob
import json

def clean(s: str, sub_pattern=re.compile('[^a-z]')) -> str:
    if not s.isascii():
        return ''
    s = s.lower()
    s = sub_pattern.sub(' ', s).strip()
    return s

clean_data: list[tuple[float, str]] = []

for fname in glob.glob('home/*.json'):
    with open(fname) as f:
        data: dict[str, dict] = json.load(f)
    for _, item in data.items():
        rating = float(item['rating'])
        quote = clean(item['quote'])
        
        words = quote.split()
        if len(words) < 5:
            continue
        quote = ' '.join(words)
        clean_data.append((rating, quote))

json.dump(clean_data, open('home_eng.json', 'w'), indent=4)