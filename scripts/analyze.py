"""Compute all numbers for the post from judge/raw_*.jsonl. Writes results/summary.json and results/per_prompt.json."""
import json, random, statistics as st, math, collections
J={'deepseek':'judge/raw_deepseek-v4.1-flash.jsonl','qwen':'judge/raw_qwen3.8-flash.jsonl'}
tasks={t['key']:t for t in map(json.loads,open('judge/tasks.jsonl'))}
R={j:{d['key']:d for d in map(json.loads,open(f))} for j,f in J.items()}
def lp(t,s):  # WildBench length penalty as on the SmolTalkBench card
    return min(s,0.5) if len(t['reply'])>len(t['reference'])+500 else s
def boot(xs,n=2000,seed=0):
    r=random.Random(seed); m=[st.mean(r.choices(xs,k=len(xs))) for _ in range(n)]; m.sort(); return [round(m[int(.025*n)],4),round(m[int(.975*n)],4)]
def scores(j,cond,sysn,pen=True):
    out={}
    for k,d in R[j].items():
        t=tasks[k]
        if t['cond']==cond and t['system']==sysn and d['score_reply_better'] is not None:
            out[t['id']]=lp(t,d['score_reply_better']) if pen else d['score_reply_better']
    return out
S={'n_prompts':len({t['id'] for t in tasks.values()}),'judges':{}}
for j in J:
    errs=sum(1 for d in R[j].values() if d['score_reply_better'] is None)
    cost=sum((d.get('usage') or {}).get('cost',0) for d in R[j].values())
    jd={'calls':len(R[j]),'unparsed':errs,'cost_usd':round(cost,4)}
    for sysn in ['pybot','smollm']:
        o=scores(j,'orig',sysn); sw=scores(j,'swap',sysn); rp=scores(j,'rep1',sysn); oraw=scores(j,'orig',sysn,False); swraw=scores(j,'swap',sysn,False)
        ids=sorted(set(o)&set(sw)); idr=sorted(set(o)&set(rp))
        flips=[ (o[i]>0.5)!=(sw[i]>0.5) for i in ids]
        # "reply first" preference: does the reply score higher when shown first?
        first_gain=[ (oraw[i]-swraw[i]) if tasks[f'orig|{sysn}|stb|{i}']['reply_first'] else (swraw[i]-oraw[i]) for i in ids]
        jd[sysn]={
          'winrate_orig_lenpen':round(st.mean(o.values()),4),'ci95':boot(list(o.values())),
          'winrate_orig_raw':round(st.mean(oraw.values()),4),
          'winrate_swap_lenpen':round(st.mean(sw.values()),4),
          'winrate_order_avg_lenpen':round(st.mean([(o[i]+sw[i])/2 for i in ids]),4),
          'binary_wins_orig':sum(v>0.5 for v in o.values()),'n':len(o),
          'swap_flip_rate':round(st.mean(flips),4),'swap_flips':sum(flips),
          'swap_mean_abs_diff':round(st.mean(abs(o[i]-sw[i]) for i in ids),4),
          'mean_gain_when_reply_shown_first':round(st.mean(first_gain),4),
          'repeat_flip_rate':round(st.mean([(o[i]>0.5)!=(rp[i]>0.5) for i in idr]),4),'repeat_flips':sum((o[i]>0.5)!=(rp[i]>0.5) for i in idr),
          'repeat_mean_abs_diff':round(st.mean(abs(o[i]-rp[i]) for i in idr),4),
          'length_penalty_applied':sum(1 for i in o if o[i]!=oraw[i]),
        }
    for sysn in ['pybot_unrelated','smollm_unrelated']:
        f=scores(j,'fool',sysn,False); real=scores(j,'orig',sysn.split('_')[0],False)
        jd[sysn]={'n':len(f),'mean_score':round(st.mean(f.values()),4),'judged_better_than_reference':sum(v>0.5 for v in f.values()),
                  'same_prompts_real_reply_mean':round(st.mean(real[i] for i in f if i in real),4),'same_prompts_real_reply_wins':sum(real[i]>0.5 for i in f if i in real)}
    S['judges'][j]=jd
# inter-judge agreement on orig verdicts
agr={}
for sysn in ['pybot','smollm']:
    a=scores('deepseek','orig',sysn,False); b=scores('qwen','orig',sysn,False); ids=sorted(set(a)&set(b))
    A=[a[i]>0.5 for i in ids]; B=[b[i]>0.5 for i in ids]
    po=st.mean(x==y for x,y in zip(A,B)); pa=st.mean(A); pb=st.mean(B); pe=pa*pb+(1-pa)*(1-pb)
    agr[sysn]={'n':len(ids),'agreement':round(po,4),'cohen_kappa':round((po-pe)/(1-pe),4) if pe<1 else None,
               'both_say_reply_better':sum(x and y for x,y in zip(A,B)),'only_deepseek':sum(x and not y for x,y in zip(A,B)),'only_qwen':sum(y and not x for x,y in zip(A,B)),
               'pearson':round(st.correlation([a[i] for i in ids],[b[i] for i in ids]),4)}
S['inter_judge']=agr
S['total_cost_usd']=round(sum(S['judges'][j]['cost_usd'] for j in J),4)
json.dump(S,open('results/summary.json','w'),indent=1)
# per prompt for the demo site
P={str(p['id']):p for p in json.load(open('data/prompts.json')) if p['set']=='stb'}
pb={}; import glob
for f in glob.glob('outputs/pybot_shard*.jsonl'):
    for d in map(json.loads,open(f)):
        if d['set']=='stb': pb[str(d['id'])]=d
sm={str(r['id']):r['reply'] for r in json.load(open('outputs/smollm360m_stb300.json'))}
rows=[]
for i,p in sorted(P.items()):
    v={}
    for j in J:
        for sysn in ['pybot','smollm']:
            for cond in ['orig','swap','rep1']:
                d=R[j].get(f'{cond}|{sysn}|stb|{i}')
                if d: v.setdefault(j,{}).setdefault(sysn,{})[cond]=None if d['score_reply_better'] is None else round(d['score_reply_better'],4)
    rows.append(dict(id=i,language='en',dataset='HuggingFaceSwarms/SmolTalkBench (gold, test)',category=p['category'],prompt=p['instruction'],
        reference=p['reference'],reference_model=p['ref_generator'],pybot=pb[i]['reply'],pybot_module=pb[i]['module'],smollm_360m_instruct=sm[i],
        reply_shown_first_in_orig=tasks[f'orig|pybot|stb|{i}']['reply_first'],
        judge_scores_p_reply_better=v))
json.dump(rows,open('results/per_prompt.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(S,indent=1))
