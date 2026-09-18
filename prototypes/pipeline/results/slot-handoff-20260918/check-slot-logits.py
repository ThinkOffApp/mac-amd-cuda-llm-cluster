import pathlib,json,subprocess,hashlib
root=pathlib.Path('/Users/petrus/Documents/Codex/2026-09-16/che');out=root/'outputs/slot-state-logits';out.mkdir(exist_ok=True)
for name in ['gemma4','qwen27b']:
 src=root/'outputs/slot-fresh'/name;r=json.loads((src/'report.json').read_text());prompt=next(x['request']['prompt'] for x in r['records'] if x['path']=='/completion')
 d=out/name;d.mkdir(exist_ok=True);(d/'prompt.json').write_text(json.dumps(prompt)+'\n')
 cmd=[str(root/'work/gguf-state-check'),r['model'],str(src/'slots/before-final-prompt-token.bin'),str(d/'prompt.json'),'1' if name=='gemma4' else '0',str(d/'report.json')]
 with (d/'stdout.log').open('w') as stdout,(d/'stderr.log').open('w') as stderr:code=subprocess.run(cmd,stdout=stdout,stderr=stderr,timeout=180).returncode
 (d/'run.json').write_text(json.dumps({'command':cmd,'exit_code':code,'source_sha256':hashlib.sha256((root/'work/tp-optimization/prototypes/pipeline/gguf_state_check.cpp').read_bytes()).hexdigest()},indent=2)+'\n')
 print(name,code,flush=True)
 if (d/'report.json').exists():
  rep=json.loads((d/'report.json').read_text());print(json.dumps({'valid':rep['valid'],'steps':len(rep['steps']),'max_error':max(x['max_abs_error'] for x in rep['steps'])}),flush=True)
