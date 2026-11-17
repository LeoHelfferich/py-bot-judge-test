# Results (17 Nov 2026)

- `summary.json`: all aggregate numbers (both judges, position swap, repeat, unrelated replies, inter-judge agreement, cost).
- `per_prompt.json`: 300 SmolTalkBench prompts with reference, PyBot reply, SmolLM-360M-Instruct reply and every judge score
  (`judge_scores_p_reply_better[judge][system][orig|swap|rep1]` = judge probability that the reply beats the reference; raw, without length penalty).
- `german_examples_pybot.json`: 55 German prompts (Aya Evaluation Suite, dolly_machine_translated, deu) with PyBot's reply. Not judged.

Sources: PyBot https://github.com/huggingface/pybot (Apache-2.0, commit 7d63c7f), SmolTalkBench https://huggingface.co/datasets/HuggingFaceSwarms/SmolTalkBench (Apache-2.0),
Aya Evaluation Suite https://huggingface.co/datasets/CohereLabs/aya_evaluation_suite (Apache-2.0), SmolLM-360M-Instruct https://huggingface.co/HuggingFaceTB/SmolLM-360M-Instruct.
