"""PyBot on all set=='stb' prompts in a worker process; a reply slower than 60 s kills the worker (status timeout)."""
import json, os, sys, random, time, multiprocessing as mp
def worker(conn):
    sys.path.insert(0, os.path.abspath('pybot')); os.chdir('pybot'); import pybot as cli; rt=cli.load_bot(); os.chdir('..')
    conn.send('ready')
    while True:
        text=conn.recv(); random.seed(0); t=time.perf_counter()
        try: res=rt.turn(text,{}); conn.send((" ".join(str(res['reply']).split()) or "Hello!", res.get('module'), 'ok', time.perf_counter()-t))
        except Exception as e: conn.send(("Hello!", None, 'error:'+type(e).__name__, time.perf_counter()-t))
def start():
    a,b=mp.Pipe(); p=mp.Process(target=worker,args=(b,),daemon=True); p.start(); a.recv(); return p,a
if __name__=='__main__':
    shard,n=int(sys.argv[1]),int(sys.argv[2])
    rows=[r for r in json.load(open('data/prompts.json')) if r['set']=='stb'][shard::n]
    outf=f'outputs/pybot_shard{shard}.jsonl'
    done={(d['set'],str(d['id'])) for d in map(json.loads,open(outf))} if os.path.exists(outf) else set()
    p,a=start()
    with open(outf,'a') as f:
        for r in rows:
            if (r['set'],str(r['id'])) in done: continue
            a.send(r['instruction'])
            if a.poll(60): reply,module,status,sec=a.recv()
            else: p.kill(); reply,module,status,sec='',None,'timeout',60; p,a=start()
            f.write(json.dumps(dict(set=r['set'],id=r['id'],reply=reply,module=module,status=status,seconds=round(sec,2)),ensure_ascii=False)+'\n'); f.flush()
