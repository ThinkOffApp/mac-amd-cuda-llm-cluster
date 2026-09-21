import json
import gzip
from pathlib import Path
root = Path(__file__).resolve().parent
def read(mode):
    path = root / mode / 'routing.jsonl'
    text = path.read_text() if path.exists() else gzip.decompress(path.with_suffix('.jsonl.gz').read_bytes()).decode()
    rows = [json.loads(line) for line in text.splitlines()]
    result = {}
    for row in rows:
        key = (row['prompt'], row['step'], row['name'])
        if key in result:
            raise RuntimeError(f'duplicate trace key {key}')
        result[key] = row
    return result
a, b = read('tensor-single'), read('tensor-local')
if a.keys() != b.keys():
    raise RuntimeError(f'trace keys differ: {len(a)} vs {len(b)}')
differences = []
for key in a:
    x, y = a[key], b[key]
    if x['shape'] != y['shape']:
        raise RuntimeError(f'shape mismatch {key}')
    if x['values'] == y['values']:
        continue
    entry = {'prompt': key[0], 'step': key[1], 'name': key[2], 'max_abs_difference': max(abs(i-j) for i,j in zip(x['values'], y['values']))}
    if 'topk' in key[2] or 'top_k' in key[2]:
        width = x['shape'][0]
        entry['different_selection_sets'] = any(set(x['values'][i:i+width]) != set(y['values'][i:i+width]) for i in range(0,len(x['values']),width))
        entry['single'] = x['values']
        entry['two'] = y['values']
    differences.append(entry)
summary = {'matched_keys': len(a), 'differences': differences}
(root / 'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'matched_keys':len(a), 'selection_differences':[x for x in differences if x.get('different_selection_sets')], 'float_difference_count':sum('different_selection_sets' not in x for x in differences)},indent=2))
