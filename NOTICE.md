# Notices

This repository contains scripts and results. Third-party material is used as follows:

- **PyBot** by Hugging Face, https://github.com/huggingface/pybot, Apache License 2.0.
  Not included here. It is cloned at commit `7d63c7f` to run it. `outputs/` and `results/` contain PyBot's replies.
- **SmolTalkBench** by Hugging Face, https://huggingface.co/datasets/HuggingFaceSwarms/SmolTalkBench, Apache License 2.0.
  `data/prompts.json`, `judge/tasks.jsonl` and `results/` contain prompts and reference answers from the `gold` split.
- **Aya Evaluation Suite** by Cohere Labs, https://huggingface.co/datasets/CohereLabs/aya_evaluation_suite, Apache License 2.0
  (`dolly_machine_translated`, derived from databricks-dolly-15k, CC BY-SA 3.0). German prompts are in `data/prompts.json`
  and `results/german_examples_pybot.json`.
- **AlpacaEval** annotator prompt `alpaca_eval_clf_gpt4_turbo`, https://github.com/tatsu-lab/alpaca_eval, Apache License 2.0
  (`judge/alpaca_eval_clf.txt`).
- **SmolLM-360M-Instruct** by Hugging Face, https://huggingface.co/HuggingFaceTB/SmolLM-360M-Instruct, Apache License 2.0.
  Model outputs are in `outputs/` and `results/`.
