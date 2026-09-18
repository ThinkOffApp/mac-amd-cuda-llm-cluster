import sys,json,datetime
d=json.load(sys.stdin); now=datetime.datetime.now(datetime.timezone.utc)
for k,v in d.get("devices",{}).items():
    ts=v.get("updated_at"); age="?"
    if ts: age=int((now-datetime.datetime.fromisoformat(ts.replace("Z","+00:00"))).total_seconds())
    print(f"{k:22s} lan_ip={v.get('lan_ip')} ctx={v.get('context')} status={v.get('status')} age={age}s ttl={v.get('ttl_sec') or v.get('ttl')} src={v.get('source_device') or v.get('source')}")
ag=d.get("agents")
print("agents:", list(ag.keys()) if isinstance(ag,dict) else ag)
