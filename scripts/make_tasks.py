"""Judge tasks, English only (300 SmolTalkBench prompts).
orig = reply/reference order fixed per prompt (sha256 of id, like AlpacaEval's fixed per-prompt order)
swap = same pair, order reversed (position bias)
rep1 = orig repeated (self-consistency)
fool = 50 prompts (seed 42) where the reply is the reply to a different prompt (fluent but unrelated)."""
import json, hashlib, random, glob
P={(r['set'],str(r['id'])):r for r in json.load(open('data/prompts.json')) if r['set']=='stb'}
pb={}
for f in glob.glob('outputs/pybot_shard*.jsonl'):
    for l in open(f):
        d=json.loads(l)
        if d['set']=='stb': pb[(d['set'],str(d['id']))]=d['reply']
sm={(r['set'],str(r['id'])):r['reply'] for r in json.load(open('outputs/smollm360m_stb300.json'))}
assert set(P)<=set(pb) and set(P)<=set(sm), (len(P),len(pb),len(sm))
OUT={'pybot':pb,'smollm':sm}
def first(k): return int(hashlib.sha256(f'{k[0]}-{k[1]}'.encode()).hexdigest(),16)%2==0
T=[]
def add(cond,sysname,k,reply,rf):
    p=P[k]; T.append(dict(key=f'{cond}|{sysname}|{k[0]}|{k[1]}',cond=cond,system=sysname,set=k[0],id=k[1],
        instruction=p['instruction'],reference=p['reference'],reply=reply,reply_first=rf))
keys=sorted(P)
for sysname,o in OUT.items():
    for k in keys:
        add('orig',sysname,k,o[k],first(k)); add('swap',sysname,k,o[k],not first(k)); add('rep1',sysname,k,o[k],first(k))
pick=random.Random(42).sample(keys,50)
for i,k in enumerate(pick):
    donor=pick[(i+1)%50]
    add('fool','pybot_unrelated',k,OUT['pybot'][donor],first(k))
    add('fool','smollm_unrelated',k,OUT['smollm'][donor],first(k))
with open('judge/tasks.jsonl','w') as f:
    for t in T: f.write(json.dumps(t,ensure_ascii=False)+'\n')
print(len(T),'tasks per judge')
