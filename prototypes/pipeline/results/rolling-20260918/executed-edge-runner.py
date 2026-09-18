import os,pathlib,socket,subprocess,json,tempfile
root=pathlib.Path('/Users/petrus/Documents/Codex/2026-09-16/che')
python=root/'work/tp-venv/bin/python';source=root/'work/tp-optimization/prototypes/pipeline/rolling.py'
out=root/'outputs/rolling-pipeline';text=source.read_text();summary=[]
corrupt=text.replace('            valid = generated == refs','            if run_id > args.warmups:\n                generated[0][-1] = (generated[0][-1]+1) % 50257\n            valid = generated == refs')
eos=text.replace('                    out_ids.append(choice)','                    if len(refs) == 0 and len(out_ids) == 1:\n                        choice = EOS\n                    out_ids.append(choice)')
eos=eos.replace('                i = c*b+j','                i = c*b+j\n                if i == 0 and step == 0:\n                    v = EOS')
with tempfile.TemporaryDirectory(prefix='rolling-edges-') as temp:
 for name,code,device,batch,nt in [('corruption',corrupt,'cpu',4,4),('eos',eos,'cpu',4,4),('one-token',text,'cpu',4,1),('wide',text,'cpu',16,3),('metal',text,'mps',4,4)]:
  script=pathlib.Path(temp,f'{name}.py');script.write_text(code)
  with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
  ps=[];fs=[]
  try:
   for rank in range(2):
    log=(out/f'{name}-rank{rank}.log').open('w');fs.append(log)
    env=dict(os.environ,RANK=str(rank),MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port),GPT2_DIR=str(root/'work/gpt2-review/checkpoint'))
    cmd=[str(python),str(script),'--device',device if rank==0 else 'cpu','--schedule','rolling','--batch',str(batch),'--chunks','2','--new-tokens',str(nt),'--warmups','1','--runs','2','--out',str(out/f'{name}.json')]
    ps.append(subprocess.Popen(cmd,env=env,stdout=log,stderr=log))
   codes=[p.wait(timeout=150) for p in ps]
  finally:
   for p in ps:
    if p.poll() is None:p.terminate();p.wait(timeout=10)
   for f in fs:f.close()
  d=json.loads((out/f'{name}.json').read_text())
  if name=='corruption':
   assert codes==[1,1] and not d['valid'] and not d['runs'] and d['all_run_checks']==[True,True,False,False]
   assert all('aggregate_decode_tokens_per_s' not in r for r in d['invalid_runs'])
  else:
   assert codes==[0,0] and d['valid'],(name,codes)
   for r in d['runs']:
    if name=='eos': assert [len(g) for g in r['produced_ids']]==[2,4,4,4]
    if name=='one-token':assert r['decode_tokens']==0 and r['aggregate_decode_tokens_per_s'] is None
  summary.append({'case':name,'exit_codes':codes,'passed':True});print(json.dumps(summary[-1]),flush=True)
(out/'edge-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
