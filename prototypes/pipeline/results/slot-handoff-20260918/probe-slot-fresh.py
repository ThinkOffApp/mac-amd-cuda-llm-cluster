import pathlib,json,subprocess,urllib.request,time,shutil,hashlib
root=pathlib.Path('/Users/petrus/Documents/Codex/2026-09-16/che');out=root/'outputs/slot-fresh';out.mkdir(exist_ok=True)
base='http://127.0.0.1:19084'
for name,source in [('gemma4',root/'outputs/slot-restore-full-swa/gemma4'),('qwen27b',root/'outputs/slot-restore-probe/qwen27b')]:
 prior=json.loads((source/'report.json').read_text());d=out/name;d.mkdir(exist_ok=True);slots=d/'slots';slots.mkdir(exist_ok=True)
 state=source/'slots/before-final-prompt-token.bin';target=slots/state.name;shutil.copy2(state,target)
 with target.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
 prompt=next(r['request']['prompt'] for r in prior['records'] if r['path']=='/completion');records=[]
 cmd=prior['command'].copy();cmd[cmd.index('--port')+1]='19084';cmd[cmd.index('--slot-save-path')+1]=str(slots)+'/'
 def req(path,body):
  before=time.time()
  with urllib.request.urlopen(urllib.request.Request(base+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=180) as r:answer=json.load(r)
  records.append({'at':before,'path':path,'request':body,'response':answer,'wall_s':time.time()-before});return answer
 def gen(ids):return req('/completion',{'prompt':ids,'id_slot':0,'cache_prompt':True,'n_predict':16,'temperature':0,'seed':42,'return_tokens':True})
 with (d/'server.log').open('w') as log:
  server=subprocess.Popen(cmd,stdout=log,stderr=log)
  try:
   for _ in range(180):
    if server.poll() is not None:raise RuntimeError('server exited')
    try:
     with urllib.request.urlopen(base+'/health',timeout=1) as r:
      if r.status==200:break
    except Exception:time.sleep(.5)
   else:raise RuntimeError('startup timeout')
   restored=req('/slots/0?action=restore',{'filename':target.name})
   handoff=gen(prompt)
   # A true subsequent turn extends the prompt with the returned tokens plus new user input.
   tail=req('/tokenize',{'content':' Continue the explanation clearly.','add_special':False})['tokens']
   next_prompt=prompt+handoff['tokens']+tail
   warm=gen(next_prompt)
   req('/slots/0?action=erase',{});native=gen(prompt)
   req('/slots/0?action=erase',{});native_next=gen(next_prompt)
   valid=(handoff['tokens']==native['tokens'] and warm['tokens']==native_next['tokens'] and handoff['timings']['cache_n']==2111 and handoff['timings']['prompt_n']==1)
   result={'valid':valid,'model':prior['model'],'command':cmd,'state_sha256':sha,'state_bytes':target.stat().st_size,'restored':restored,'cases':{'handoff':handoff,'warm_next_turn':warm,'native':native,'native_next_turn':native_next},'records':records}
   (d/'report.json').write_text(json.dumps(result,indent=2)+'\n')
   print(name,json.dumps({'valid':valid,'state_bytes':target.stat().st_size,'cases':{k:{'timings':v['timings'],'tokens':v['tokens']} for k,v in result['cases'].items()}}),flush=True)
  finally:
   if server.poll() is None:server.terminate()
   server.wait(timeout=10)
