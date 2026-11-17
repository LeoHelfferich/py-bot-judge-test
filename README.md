# py-bot-judge-test

A small replication test of the LLM-judge setup behind the SmolTalkBench win rate reported for
[PyBot](https://github.com/huggingface/pybot), the rule-based (no neural network) chatbot published by Hugging Face.

PyBot, SmolTalkBench and the judging protocol are Hugging Face's work. This repository only re-runs a
part of the evaluation and adds checks on the judge itself.

## What was measured

- **Prompts:** 300 English prompts from the SmolTalkBench `gold` split (150 sampled with seed 42, 150 more from the rest with seed 43).
- **Systems:** PyBot (commit `7d63c7f`, single turn, fresh memory, `random.seed(0)`) and
  [SmolLM-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM-360M-Instruct) (CPU, greedy, max 384 new tokens).
- **Judging,** following the SmolTalkBench dataset card: AlpacaEval 2.0 weighted annotator prompt
  (`alpaca_eval_clf_gpt4_turbo`), reply vs. reference answer, `max_tokens=1`, `temperature=1`, top-5 logprobs,
  score = P(reply better), order fixed per prompt, WildBench length rule (a reply more than 500 characters
  longer than the reference is capped at 0.5).
- **Two judges via OpenRouter:** `deepseek/deepseek-v4.1-flash` (provider Reka, reasoning disabled) and `qwen/qwen3.8-flash`.
- **Judge checks** (1,900 calls per judge):
  - `orig`: the protocol above
  - `swap`: same pair, order reversed
  - `rep1`: `orig` repeated
  - `fool`: 50 prompts where the reply belongs to a different prompt
- **German side check:** 55 German prompts (Aya Evaluation Suite, `dolly_machine_translated`, `deu`) were run through PyBot only. They were not judged.

## Results

All numbers come from `results/summary.json` and `results/diff_pybot_minus_smollm.json`. Win rates are mean
P(reply better) with the length rule applied. The 95% CIs are bootstrap intervals.

| | DeepSeek V4.1 Flash | Qwen3.8 Flash |
|---|---|---|
| PyBot win rate | 27.03% (22.23–32.18) | 24.08% (19.91–28.62) |
| SmolLM-360M win rate | 25.13% (22.19–28.24) | 27.74% (24.72–30.62) |
| PyBot − SmolLM (paired) | +1.90 pp (−3.70 … +7.69) | −3.65 pp (−8.56 … +1.57) |
| SmolLM win rate without length rule | 41.87% | 41.60% |
| Length rule applied (PyBot / SmolLM) | 2 / 108 | 2 / 104 |
| Swap flips (PyBot / SmolLM, of 300) | 13 / 7 | 15 / 6 |
| Repeat flips (PyBot / SmolLM, of 300) | 14 / 2 | 3 / 2 |
| Unrelated reply judged better (PyBot / SmolLM, of 50) | 1 / 0 | 1 / 0 |
| Cost (USD) | 0.0861 | 0.1255 |

On PyBot, the two judges agree on the winner for 92.0% of prompts (Cohen's κ 0.79). On SmolLM they agree for
82.7% (κ 0.64). A "flip" means the winner changed between the two runs (score above vs. below 0.5).

Both paired CIs include zero. On these 300 prompts, neither judge separates PyBot from SmolLM-360M-Instruct.
SmolLM's score depends mostly on the length rule.

German: PyBot answered none of the 55 German prompts in German (`results/german_examples_pybot.json`).

PyBot source statistics (file count, `if`/`elif` lines, quoted README claims) are in `results/pybot_repo_stats.json`.

## Limitations

- 300 of the 1,000 `gold` prompts, one seed per system, so this is not a full re-run of the benchmark.
- The original judge was called directly. Here both judges were called through OpenRouter, so provider-side
  differences (quantization, logprob handling) are possible. One provider ignored "reasoning disabled" and returned
  no usable logprobs, so that run was discarded (`judge/failed_novita_reasoning_not_disabled.jsonl`).
- SmolLM-360M-Instruct is a small baseline chosen for CPU feasibility. It is not one of the models on the SmolTalkBench leaderboard.
- `results/per_prompt.json` rounds scores to 4 decimals. Exact flip counts come from the raw `judge/raw_*.jsonl` files.

## Reproduce

```bash
pip install -r requirements.txt
git clone https://github.com/huggingface/pybot && git -C pybot checkout 7d63c7f
bash scripts/download_data.sh            # SmolTalkBench gold + Aya dolly_machine_translated
python scripts/make_prompts.py           # -> data/prompts.json
python scripts/run_pybot_stb.py 0 1      # PyBot on the 300 English prompts -> outputs/pybot_shard0.jsonl
python scripts/run_pybot.py 0 1          # optional: PyBot on the German prompts
python scripts/run_smollm.py             # -> outputs/smollm360m_stb300.json
python scripts/make_tasks.py             # -> judge/tasks.jsonl
export OPENROUTER_API_KEY=...            # your own key
python scripts/judge.py deepseek/deepseek-v4.1-flash judge/tasks.jsonl judge/raw_deepseek-v4.1-flash.jsonl --provider reka
python scripts/judge.py qwen/qwen3.8-flash judge/tasks.jsonl judge/raw_qwen3.8-flash.jsonl
python scripts/analyze.py && python scripts/diff_ci.py
python scripts/repo_stats.py             # needs the pybot clone
```

The original runs used 3 shards (`outputs/pybot_shard{0,1,2}.jsonl`). Judge outputs vary slightly between calls
because `temperature=1` is part of the protocol.

## Credits and licenses

See [NOTICE.md](NOTICE.md).
