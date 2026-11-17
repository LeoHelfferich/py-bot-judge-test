"""Paired bootstrap 95% CI for winrate(PyBot) - winrate(SmolLM), length penalty applied, per judge (orig order)."""
import json,random
R=json.load(open('results/per_prompt.json'))
def lp(reply,ref,s): return min(s,0.5) if len(reply)>len(ref)+500 else s
out={}
for j in ['deepseek','qwen']:
    d=[]
    for r in R:
        v=r['judge_scores_p_reply_better'][j]
        a=v.get('pybot',{}).get('orig'); b=v.get('smollm',{}).get('orig')
        if a is None or b is None: continue
        d.append(lp(r['pybot'],r['reference'],a)-lp(r['smollm_360m_instruct'],r['reference'],b))
    rnd=random.Random(0); m=sorted(sum(rnd.choices(d,k=len(d)))/len(d) for _ in range(5000))
    out[j]=dict(n=len(d),diff=round(sum(d)/len(d),4),ci95=[round(m[125],4),round(m[4875],4)])
print(out); json.dump(out,open('results/diff_pybot_minus_smollm.json','w'),indent=1)
