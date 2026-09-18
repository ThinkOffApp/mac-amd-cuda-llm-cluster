import subprocess,pathlib,socket,time,json,os,hashlib
root=pathlib.Path('/Users/petrus/Documents/Codex/2026-09-16/che');engine=root/'work/llama-cpp-tp';out=root/'outputs/gguf-contexts-no-blas';out.mkdir(exist_ok=True)
env=dict(os.environ,GGUF_PROBE_NO_BLAS='1');env.pop('GGML_META_PROFILE',None);env.pop('GGML_BACKEND_PATH',None)
models=[('gemma4',root/'work/models/gemma-4-e4b/gemma-4-E4B-it-Q4_K_M.gguf',root/'outputs/heterogeneous-tp/engine-gemma-e4b/solo'),('qwen27b',pathlib.Path('/Volumes/t705/ModelArchive/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf'),root/'outputs/heterogeneous-tp/engine-correctness/solo')]
for name,model,ref in models:
 for mode in ['solo','layer']:
  folder=out/f'{name}-{mode}';folder.mkdir(exist_ok=True);server=None
  with (folder/'server.log').open('w') as serverlog:
   try:
    if mode=='layer':
     server=subprocess.Popen([str(engine/'build-tp/bin/ggml-rpc-server'),'-d','MTL0','-H','127.0.0.1','-p','29591','-t','4'],env=env,stdout=serverlog,stderr=serverlog)
     for _ in range(100):
      if server.poll() is not None:raise RuntimeError('server failed')
      try:
       with socket.create_connection(('127.0.0.1',29591),timeout=.1):pass
       break
      except OSError:time.sleep(.1)
     else:raise RuntimeError('server startup timeout')
    cmd=[str(root/'work/gguf-contexts'),str(model),mode,'127.0.0.1:29591',str(ref),str(folder/'report.json')]
    with (folder/'stdout.log').open('w') as stdout,(folder/'stderr.log').open('w') as stderr:
     try:code=subprocess.run(cmd,env=env,stdout=stdout,stderr=stderr,timeout=240).returncode
     except subprocess.TimeoutExpired:code='timeout'
    (folder/'run.json').write_text(json.dumps({'command':cmd,'exit_code':code,'source_sha256':hashlib.sha256((root/'work/tp-optimization/prototypes/pipeline/gguf_contexts.cpp').read_bytes()).hexdigest(),'engine_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=engine,text=True).strip()},indent=2)+'\n')
    print(name,mode,code,flush=True)
    if (folder/'report.json').exists():
     r=json.loads((folder/'report.json').read_text());print(json.dumps({'valid':r['valid'],'runs':[{k:row[k] for k in ['schedule','warmup','valid','wall_s','generated_tokens','max_abs_logit_error','outside_tolerance']} for row in r['runs']]}),flush=True)
   finally:
    if server:
     if server.poll() is None:server.terminate()
     server.wait(timeout=10)
