"""Build the fixed prompt set data/prompts.json.
- stb: 300 SmolTalkBench `gold` prompts = 150 sampled with seed 42, plus 150 more from the remaining rows with seed 43.
- dolly_de: the 200 German prompts of Aya Evaluation Suite `dolly_machine_translated` (only used for qualitative PyBot examples).
Needs data/smoltalkbench_gold.parquet and data/aya_dolly_mt.parquet (see scripts/download_data.sh)."""
import pandas as pd, json
s = pd.read_parquet('data/smoltalkbench_gold.parquet')
first = s.sample(n=150, random_state=42)
rest = s[~s.id.isin(set(first.id))].sample(n=150, random_state=43)
rows = [dict(set='stb', id=r.id, instruction=r.instruction, reference=r.output, ref_generator=r.generator, category=r.category)
        for part in (first, rest) for r in part.itertuples()]
d = pd.read_parquet('data/aya_dolly_mt.parquet')
de = d[d.language == 'deu'].sort_values('id')
rows += [dict(set='dolly_de', id=int(r.id), instruction=r.inputs, reference=r.targets,
              ref_generator='databricks-dolly-15k (human); DE via NLLB', category='dolly') for r in de.itertuples()]
json.dump(rows, open('data/prompts.json', 'w'), ensure_ascii=False, indent=1)
print({k: sum(r['set'] == k for r in rows) for k in ['stb', 'dolly_de']})
