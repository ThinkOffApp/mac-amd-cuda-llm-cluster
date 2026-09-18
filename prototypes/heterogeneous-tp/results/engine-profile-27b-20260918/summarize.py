import json, statistics, gzip
from pathlib import Path
root = Path(__file__).resolve().parent
report = json.loads((root/'tensor/report.json').read_text())
assert report['valid'], 'profile correctness gate failed'
steps=[]
current=None
log = root/'tensor/stderr.log'
text = log.read_text() if log.exists() else gzip.decompress(log.with_suffix('.log.gz').read_bytes()).decode()
for line in text.splitlines():
    if line.startswith('ENGINE_STEP_BEGIN '):
        assert current is None
        current=json.loads(line.split(' ',1)[1]);current['graphs']=[]
    elif line.startswith('META_PROFILE '):
        assert current is not None, 'profile row outside decode'
        row=json.loads(line.split(' ',1)[1])
        assert row['ranks']==2
        assert row['collectives']==row['subgraphs']-1
        assert all(row[k]>=0 for k in ['prepare_us','initial_wait_us','compute_us','collective_us','total_us','collective_input_bytes'])
        assert sum(row[k] for k in ['prepare_us','initial_wait_us','compute_us','collective_us'])<=row['total_us']
        current['graphs'].append(row)
    elif line.startswith('ENGINE_STEP_END '):
        assert current is not None
        end=json.loads(line.split(' ',1)[1])
        assert (end['prompt'],end['step'])==(current['prompt'],current['step'])
        current['wall_us']=end['wall_us']
        assert current['graphs']
        assert sum(r['total_us'] for r in current['graphs'])<=current['wall_us']
        steps.append(current);current=None
assert current is None
assert len(steps)==sum(len(c['chosen_ids']) for c in report['cases'])
summary={}
for phase in ['prefill','decode']:
    rows=[s for s in steps if (s['step']==0)==(phase=='prefill')]
    totals={k:sum(g[k] for s in rows for g in s['graphs']) for k in ['prepare_us','initial_wait_us','compute_us','collective_us','collectives','collective_input_bytes','total_us']}
    wall=sum(s['wall_us'] for s in rows)
    summary[phase]={'steps':len(rows),'wall_us':wall,'median_step_us':statistics.median(s['wall_us'] for s in rows),**totals,'collective_share_of_decode_call_wall':totals['collective_us']/wall,'unattributed_us':wall-sum(totals[k] for k in ['prepare_us','initial_wait_us','compute_us','collective_us'])}
result={'scope':'Instrumented same-physical-GPU Metal + loopback Metal RPC. Synchronization alters scheduling. No pure-wire or physical-pair performance claim. Collective input bytes sum local participant tensors, not network bytes.','correctness_valid':report['valid'],'phases':summary,'steps':steps}
(root/'profile.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(summary,indent=2))
