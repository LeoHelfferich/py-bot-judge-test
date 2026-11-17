"""Counts in the PyBot clone (commit pinned) + verbatim Hugging Face claims from its README. Writes results/pybot_repo_stats.json."""
import subprocess, pathlib, re, json, random
root=pathlib.Path('pybot'); commit=subprocess.check_output(['git','-C','pybot','rev-parse','HEAD'],text=True).strip()
files=sorted((root/'bot').rglob('*.py'))
lines=0; iflines=[]
for f in files:
    txt=f.read_text(encoding='utf-8',errors='replace').split('\n')
    lines+=len(txt)-(1 if txt and txt[-1]=='' else 0)   # same as wc -l for files ending in newline
    for l in txt:
        s=l.strip()
        if re.match(r'^(if|elif) .{8,46}:$',s) and '"' not in s and "'" not in s: iflines.append(s)
n_if=sum(1 for f in files for l in f.read_text(encoding='utf-8',errors='replace').split('\n') if re.match(r'^\s*(if|elif)\b',l))
readme=(root/'README.md').read_text()
claims={k:re.search(p,readme).group(0) for k,p in {'agents':r'a swarm of over 1,000 agents','lines':r'around 640k lines of code','files':r'528 Python files and about 640,000 lines','winrate':r'climbed from 0% to 26%'}.items()}
random.Random(7).shuffle(iflines)
out=dict(commit=commit,py_files=len(files),py_lines=lines,if_elif_lines=n_if,hf_readme_claims=claims,sample_if_lines=iflines[:400])
json.dump(out,open('results/pybot_repo_stats.json','w'),indent=1,ensure_ascii=False)
print({k:v for k,v in out.items() if k!='sample_if_lines'}); print(iflines[:8])
