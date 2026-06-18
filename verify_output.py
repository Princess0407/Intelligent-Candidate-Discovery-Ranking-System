import csv, re, collections, sys

sys.stdout.reconfigure(encoding='utf-8')

with open('submission.csv', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
rows.sort(key=lambda r: int(r['rank']))

# 1. Bracket check
brackets = [
    (r['rank'], r['candidate_id'], re.findall(r'\[(?!Rank \d)', r['reasoning']))
    for r in rows if re.search(r'\[(?!Rank \d)', r['reasoning'])
]
print(f'[N] or unexpected brackets: {len(brackets)}')

# 2. Concern sentence distribution in top 10
print('\nTop-10 concern sentences:')
for r in rows[:10]:
    concern = ''
    m = re.search(r'Primary concern: (.+?)\.?\s*$', r['reasoning'])
    if m:
        concern = m.group(1)
    rank = r['rank']
    cid = r['candidate_id']
    print(f'  Rank {rank:>2}  {cid}  -> {concern}')

# 3. Concern frequency across all 100
concerns_all = []
for r in rows:
    m = re.search(r'Primary concern: (.+?)\.?\s*$', r['reasoning'])
    if m:
        concerns_all.append(m.group(1))

print(f'\nConcern frequency across all 100 ({len(concerns_all)} rows with concern):')
for concern, cnt in collections.Counter(concerns_all).most_common():
    print(f'  {cnt:>3}x  {concern}')
