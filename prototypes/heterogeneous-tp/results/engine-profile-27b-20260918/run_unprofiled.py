import json,subprocess,time,socket,os,hashlib
from pathlib import Path
base=Path('/Users/petrus/Documents/Codex/2026-09-16/che')
wd=base/'work/llama-cpp-tp'
out=base/'outputs/heterogeneous-tp/engine-profile-27b-off'
out.mkdir(parents=True,exist_ok=True)
model='/Volumes/t705/ModelArchive/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf'
env=dict(os.environ,GGML_BACKEND_PATH=str(wd/'build-tp/bin'),GGML_META_PROFILE='0')
for mode in ['tensor']:
 dest=out/mode;dest.mkdir(exist_ok=True)
 server=None
 with open(dest/'server.log','w') as log:
  try:
   if mode!='solo':
    server=subprocess.Popen([str(wd/'build-tp/bin/ggml-rpc-server'),'-d','MTL0','-H','127.0.0.1','-p','29591','-t','4'],stdout=log,stderr=log,env=env)
    for _ in range(150):
     if server.poll() is not None:raise RuntimeError('server exited')
     try:
      with socket.create_connection(('127.0.0.1',29591),timeout=.1):pass
      break
     except OSError:time.sleep(.1)
    else:raise RuntimeError('server not ready')
   cmd=[str(base/'work/engine-profile-correctness'),model,mode,'127.0.0.1:29591',str(dest),str(base/'outputs/heterogeneous-tp/engine-correctness/solo')]
   start=time.time()
   with open(dest/'stdout.log','w') as stdout,open(dest/'stderr.log','w') as stderr:
    try:code=subprocess.run(cmd,stdout=stdout,stderr=stderr,env=env,cwd=wd,timeout=300).returncode
    except subprocess.TimeoutExpired:code='timeout'
   (dest/'run.json').write_text(json.dumps(dict(command=cmd,returncode=code,wall_s=time.time()-start,engine_commit='bdcbaaf6e7520b68c8c60ff724c67409970d70e1',source_sha256=hashlib.sha256((base/'outputs/heterogeneous-tp/engine-profile-27b/profile_correctness.cpp').read_bytes()).hexdigest(),rpc_patch_sha256=hashlib.sha256(subprocess.check_output(['git','diff'],cwd=wd)).hexdigest()),indent=2))
   print(mode,code,flush=True)
   if (dest/'report.json').exists():
    r=json.loads((dest/'report.json').read_text());print(json.dumps({'valid':r['valid'],'cases':[{k:c[k] for k in ['tokens_match','max_abs_logit_error','logits_within_tolerance','min_reference_top2_margin']} for c in r['cases']]}),flush=True)
   if mode=='solo' and code!=0:break
  finally:
   if server:
    if server.poll() is None:server.terminate()
    server.wait(timeout=10)
