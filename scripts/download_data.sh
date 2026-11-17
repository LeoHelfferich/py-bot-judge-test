#!/usr/bin/env bash
set -euo pipefail
mkdir -p data
curl -L -o data/smoltalkbench_gold.parquet https://huggingface.co/datasets/HuggingFaceSwarms/SmolTalkBench/resolve/main/gold/test-00000-of-00001.parquet
curl -L -o data/aya_dolly_mt.parquet https://huggingface.co/datasets/CohereLabs/aya_evaluation_suite/resolve/main/dolly_machine_translated/test-00000-of-00001.parquet
