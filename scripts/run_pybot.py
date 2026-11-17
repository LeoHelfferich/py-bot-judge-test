"""Run PyBot (huggingface/pybot @7d63c7f) on every prompt: single turn, fresh memory, random.seed(0).
A reply that takes longer than 60 s is recorded as timeout (reply = ""). Usage: run_pybot.py SHARD NSHARDS"""
import json, sys, random, time, os, signal
sys.path.insert(0, os.path.abspath('pybot'))
os.chdir('pybot'); import pybot as cli; rt = cli.load_bot(); os.chdir('..')
shard, n = int(sys.argv[1]), int(sys.argv[2])
rows=json.load(open('data/prompts.json'))[shard::n]
outf=f'outputs/pybot_shard{shard}.jsonl'
done=set()
if os.path.exists(outf): done={(d['set'],str(d['id'])) for d in map(json.loads,open(outf))}
class TO(Exception): pass
def h(*a): raise TO()
signal.signal(signal.SIGALRM, h)
with open(outf,'a') as f:
    for r in rows:
        if (r['set'],str(r['id'])) in done: continue
        random.seed(0); t=time.perf_counter(); signal.alarm(60)
        try:
            res=rt.turn(r['instruction'], {}); reply=" ".join(str(res['reply']).split()) or "Hello!"; module=res.get('module'); status='ok'
        except TO: reply, module, status = "", None, 'timeout'
        except Exception as e: reply, module, status = "Hello!", None, 'error:'+type(e).__name__
        finally: signal.alarm(0)
        f.write(json.dumps(dict(set=r['set'], id=r['id'], reply=reply, module=module, status=status, seconds=round(time.perf_counter()-t,2)), ensure_ascii=False)+'\n'); f.flush()
