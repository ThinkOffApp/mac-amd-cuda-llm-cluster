import argparse, os, sys, subprocess, json, socket
from pathlib import Path
parser = argparse.ArgumentParser(description="Local CPU integration check; requires GPT2_DIR")
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
if not os.environ.get("GPT2_DIR"):
 parser.error("Set GPT2_DIR to the pinned checkpoint directory")
wd = Path(__file__).resolve().parent
py = sys.executable
out = args.out.resolve()
out.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, WORLD_SIZE='2', MASTER_ADDR='127.0.0.1')
for prompt in [128,512]:
 for head in ['baseline','last-root']:
  for mode in ['solo','tp']:
   name=f'{mode}-p{prompt}-{head}'
   with socket.socket() as sock:
    sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
   env['MASTER_PORT']=str(port)
   cmd=[py,'bench.py','--mode',mode,'--device','cpu','--prompt-tokens',str(prompt),'--new-tokens','8','--runs','2','--warmups','1','--generation-head',head,'--profile','--out',str(out/(name+'.json'))]
   processes=[]
   handles=[]
   for rank in range(2 if mode=='tp' else 1):
    log=open(out/f'{name}-rank{rank}.log','w');handles.append(log)
    processes.append(subprocess.Popen(cmd,cwd=wd,env=dict(env,RANK=str(rank)),stdout=log,stderr=log))
   try:
    codes=[p.wait(timeout=180) for p in processes]
   finally:
    for process in processes:
     if process.poll() is None:
      process.kill()
      process.wait()
    for h in handles:
     h.close()
   assert codes==[0]*len(codes),(name,codes)
   report=json.loads((out/(name+'.json')).read_text())
   ranks=report.get('ranks',[report])
   for r in ranks:
    assert r['valid'] and r['all_timed_runs_matched_reference']
    assert r['timings']==[]
    assert len(r['profile_runs'])==2
    assert r['profile_runs'][0]['phases']['prefill']['attention_compute']['calls']==12
    assert r['profile_runs'][1]['phases']['prefill']['attention_compute']['calls']==12
    stats=r['profile_runs'][1]['phases']
    if mode=='tp':
     assert stats['prefill']['collective']['calls']==24
     assert stats['decode']['collective']['calls']==24*7
     assert stats['prefill']['token_broadcast']['calls']==1
     assert stats['decode']['token_broadcast']['calls']==7
     for phase,count,tokens in [('prefill',1,prompt),('decode',7,1)]:
      expected=24*count*tokens*768*4
      send=stats[phase]['socket_send']['payload_bytes']
      recv=stats[phase]['socket_receive']['payload_bytes']
      assert send==expected+(4*count if r['rank']==0 else 0),(phase,send,expected)
      assert recv==expected+(4*count if r['rank']==1 else 0),(phase,recv,expected)
   print(name,'PASS',flush=True)
print('All outputs, profile counts and wire payload sizes verified.',flush=True)
