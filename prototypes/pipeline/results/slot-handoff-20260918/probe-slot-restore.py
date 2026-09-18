import pathlib,json,subprocess,urllib.request,time,hashlib
root=pathlib.Path('/Users/petrus/Documents/Codex/2026-09-16/che');out=root/'outputs/slot-restore-probe';out.mkdir(exist_ok=True)
models=[('gemma4',root/'work/models/gemma-4-e4b/gemma-4-E4B-it-Q4_K_M.gguf'),('qwen27b',pathlib.Path('/Volumes/t705/ModelArchive/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf'))]
base='http://127.0.0.1:19083'
for name,model in models:
 d=out/name;d.mkdir(exist_ok=True);slots=d/'slots';slots.mkdir(exist_ok=True);records=[]
 cmd=[str(root/'work/llama-cpp-tp/build-tp/bin/llama-server'),'-m',str(model),'-ngl','999','-c','4096','-np','1','-b','512','-ub','512','-t','4','--host','127.0.0.1','--port','19083','--slot-save-path',str(slots)+'/', '--no-webui','--log-verbosity','4','--cache-ram','0']
 def req(path,body=None):
  before=time.time();data=None if body is None else json.dumps(body).encode()
  with urllib.request.urlopen(urllib.request.Request(base+path,data=data,headers={'Content-Type':'application/json'}),timeout=120) as r:answer=json.load(r)
  records.append({'at':before,'path':path,'request':body,'response':answer,'wall_s':time.time()-before});return answer
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
   toks=req('/tokenize',{'content':('A careful experiment records the prompt, the cache state, and every generated token. '*400),'add_special':True})['tokens'][:2112]
   assert len(toks)==2112
   def generate(ids,n=4):return req('/completion',{'prompt':ids,'id_slot':0,'cache_prompt':True,'n_predict':n,'temperature':0,'seed':42,'return_tokens':True})
   def erase():return req('/slots/0?action=erase',{})
   def save(label):return req('/slots/0?action=save',{'filename':label+'.bin'})
   def restore(label):return req('/slots/0?action=restore',{'filename':label+'.bin'})
   cold=generate(toks);save('after-generation');erase();restore('after-generation');replay=generate(toks)
   erase();producer=generate(toks[:-1],1);saved=save('before-final-prompt-token');erase();restored=restore('before-final-prompt-token');handoff=generate(toks)
   result={'model':str(model),'engine_commit':'bdcbaaf6e7520b68c8c60ff724c67409970d70e1','command':cmd,'cases':{'cold':cold,'replay_after_generation':replay,'producer_prefix':producer,'handoff_append_final_prompt_token':handoff},'save':saved,'restore':restored,'records':records}
   (d/'report.json').write_text(json.dumps(result,indent=2)+'\n')
   print(name,json.dumps({k:{'timings':v.get('timings'),'tokens':v.get('tokens'),'content':v.get('content')} for k,v in result['cases'].items()}),flush=True)
  finally:
   if server.poll() is None:server.terminate()
   server.wait(timeout=10)
