import os,pathlib,socket,subprocess,json,sys
import argparse,sys
ap=argparse.ArgumentParser()
ap.add_argument('--model-dir',type=pathlib.Path,required=True)
ap.add_argument('--out',type=pathlib.Path,required=True)
args=ap.parse_args()
python=pathlib.Path(sys.executable)
source=pathlib.Path(__file__).resolve().parent/'rolling.py'
out=args.out;out.mkdir(parents=True,exist_ok=True)
summary=[]
for schedule in ['alternating','barrier','rolling']:
 with socket.socket() as s:
  s.bind(('127.0.0.1',0));port=s.getsockname()[1]
 procs=[];files=[]
 try:
  for rank in range(2):
   log=(out/f'{schedule}-rank{rank}.log').open('w');files.append(log)
   env=dict(os.environ,RANK=str(rank),MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port),GPT2_DIR=str(args.model_dir.resolve()))
   cmd=[str(python),str(source),'--schedule',schedule,'--batch','4','--chunks','2','--new-tokens','8','--warmups','1','--runs','2','--out',str(out/f'{schedule}.json')]
   procs.append(subprocess.Popen(cmd,env=env,stdout=log,stderr=log))
  codes=[p.wait(timeout=150) for p in procs]
 finally:
  for p in procs:
   if p.poll() is None:p.terminate();p.wait(timeout=10)
  for f in files:f.close()
 assert codes==[0,0],(schedule,codes)
 report=json.loads((out/f'{schedule}.json').read_text())
 assert report['valid'] and all(report['all_run_checks'])
 for run in report['runs']:
  assert run['produced_ids']==report['reference_ids']
  events=run['host_event_order']
  if schedule=='rolling':
   assert events.index(['send',0,1]) < events.index(['recv',1,0])
  else:
   assert events.index(['send',0,1]) > events.index(['recv',1,0])
 summary.append({'schedule':schedule,'exit_codes':codes,'valid':True,'ordering_valid':True})
 print(json.dumps(summary[-1]),flush=True)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
