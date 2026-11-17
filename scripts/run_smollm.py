"""Run HuggingFaceTB/SmolLM-360M-Instruct on CPU, greedy, max 384 new tokens, chat template, single turn."""
import json, torch, time, sys
from transformers import AutoTokenizer, AutoModelForCausalLM
M='HuggingFaceTB/SmolLM-360M-Instruct'
torch.manual_seed(0); torch.set_num_threads(6)
tok=AutoTokenizer.from_pretrained(M); tok.padding_side='left'
if tok.pad_token is None: tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.float32).eval()
rows=json.load(open('data/prompts.json'))
order = [r for r in rows if r['set']=='stb']
lim = int(sys.argv[1]) if len(sys.argv)>1 else None
if lim: order=order[:lim]
out=[]; B=8; t0=time.time()
for i in range(0,len(order),B):
    batch=order[i:i+B]
    texts=[tok.apply_chat_template([{'role':'user','content':r['instruction']}], tokenize=False, add_generation_prompt=True) for r in batch]
    enc=tok(texts, return_tensors='pt', padding=True)
    with torch.no_grad():
        g=model.generate(**enc, max_new_tokens=384, do_sample=False, pad_token_id=tok.pad_token_id)
    for r,seq in zip(batch, g[:, enc['input_ids'].shape[1]:]):
        out.append(dict(set=r['set'], id=r['id'], reply=tok.decode(seq, skip_special_tokens=True).strip()))
    print(i+len(batch), round(time.time()-t0), flush=True)
    json.dump(out, open('outputs/smollm360m_stb300.json' if not lim else 'outputs/smollm_test.json','w'), ensure_ascii=False, indent=1)
