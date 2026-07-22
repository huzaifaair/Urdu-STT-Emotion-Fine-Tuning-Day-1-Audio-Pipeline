import json
recs = [json.loads(l) for l in open('processed/manifest.jsonl', encoding='utf-8')]
fresh = [r for r in recs if r.get('source_type') == 'fresh_recording']
print('Total fresh_recording segments:', len(fresh))
from collections import Counter
print('Emotion distribution:', Counter(r.get('emotion') for r in fresh))
print('Accent distribution:', Counter(r.get('accent') for r in fresh))
print()
print('Sample segment IDs and their emotion:')
for r in fresh[:10]:
    print(r.get('segment_id'), '->', r.get('emotion'), '/', r.get('accent'))