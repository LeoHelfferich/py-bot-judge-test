"""Pairwise judge, AlpacaEval 2.0 weighted annotator (alpaca_eval_clf_gpt4_turbo prompt),
as described on the SmolTalkBench card: reply vs reference, order fixed per prompt,
score = P(reply better) from top-5 logprobs of the first token ('m' = output_1, 'M' = output_2).
Usage: python scripts/judge.py MODEL TASKFILE OUTFILE [--dry]
OpenRouter key is read from env OPENROUTER_API_KEY and never printed."""
import json, os, sys, math, hashlib, time, re, concurrent.futures as cf, requests
PROVIDER={'require_parameters':True}
TPL=open('judge/alpaca_eval_clf.txt').read()
def to_messages(instr, o1, o2):
    t=TPL.replace('{instruction}',instr).replace('{output_1}',o1).replace('{output_2}',o2)
    sys_part=re.search(r'<\|im_start\|>system\n(.*?)<\|im_end\|>',t,re.S).group(1).strip()
    usr_part=re.search(r'<\|im_start\|>user\n(.*?)<\|im_end\|>',t,re.S).group(1).strip()
    return [{'role':'system','content':sys_part},{'role':'user','content':usr_part}]
def call(model, msgs, tries=6):
    body={'model':model,'messages':msgs,'max_tokens':1,'temperature':1,'logprobs':True,'top_logprobs':5,
          'reasoning':{'enabled':False}, 'provider':PROVIDER}
    for k in range(tries):
        try:
            r=requests.post('https://openrouter.ai/api/v1/chat/completions',json=body,timeout=120,
                headers={'Authorization':'Bearer '+os.environ['OPENROUTER_API_KEY']})
            j=r.json()
            if 'choices' in j: return j
            err=j.get('error')
        except Exception as e: err=str(e)
        time.sleep(2*(k+1))
    return {'error':str(err)}
def p_first(j):
    """P('m') normalised over m/M, from top logprobs of first token."""
    try:
        lp=j['choices'][0]['logprobs']['content'][0]['top_logprobs']
    except Exception: return None, None
    pm=sum(math.exp(x['logprob']) for x in lp if x['token'].strip()=='m')
    pM=sum(math.exp(x['logprob']) for x in lp if x['token'].strip()=='M')
    if pm+pM==0: return None, j['choices'][0]['message'].get('content')
    return pm/(pm+pM), j['choices'][0]['message'].get('content')
def run(model, tasks, outfile, workers=8):
    done={}
    if os.path.exists(outfile):
        for l in open(outfile): d=json.loads(l); done[d['key']]=d
    todo=[t for t in tasks if t['key'] not in done]
    def one(t):
        # reply_first: reply is output_1 ('m'); else reference is output_1
        o1,o2=(t['reply'],t['reference']) if t['reply_first'] else (t['reference'],t['reply'])
        j=call(model,to_messages(t['instruction'],o1,o2))
        p,tok=p_first(j)
        score=None if p is None else (p if t['reply_first'] else 1-p)
        return dict(key=t['key'],model=model,score_reply_better=score,first_token=tok,
                    usage=j.get('usage'),error=j.get('error'))
    with open(outfile,'a') as f, cf.ThreadPoolExecutor(workers) as ex:
        for i,res in enumerate(ex.map(one,todo)):
            f.write(json.dumps(res,ensure_ascii=False)+'\n'); f.flush()
            if i%100==0: print(i,len(todo),flush=True)
if __name__=='__main__':
    model,taskfile,outfile=sys.argv[1:4]
    if '--provider' in sys.argv: PROVIDER.update(order=[sys.argv[sys.argv.index('--provider')+1]], allow_fallbacks=False)
    tasks=[json.loads(l) for l in open(taskfile)]
    if '--limit' in sys.argv: tasks=tasks[:int(sys.argv[sys.argv.index('--limit')+1])]
    if '--dry' in sys.argv:
        chars=sum(len(json.dumps(to_messages(t['instruction'],t['reply'],t['reference']),ensure_ascii=False)) for t in tasks)
        print(len(tasks),'calls, ~',chars//3.5,'input tokens (chars/3.5)'); sys.exit()
    run(model,tasks,outfile)
