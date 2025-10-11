# Code for "Complexity-aware fine-tuning" paper

General-purpose Large Language Models (LLMs) are frequently fine-tuned through supervised fine-tuning (SFT) to enhance performance in specific domains. Better results can be achieved by distilling the chain-of-thought of a larger model at the cost of numerous expensive calls and a much greater amount of data.
We propose a novel blueprint for efficient fine-tuning that uses reasoning only for complex data identified by entropy. Specifically, across two small open models ($\approx 3B$) we split the training data into complexity categories by a single token answer entropy (ROC AUC $0.73$), fine-tune large language models (LLMs) via SFT and distillation, and show that our pipeline significantly outperforms the standard SFT approach ($0.58$ vs $0.45$ average accuracy) and outperforms the distillation approach ($0.58$ vs $0.56$ average accuracy) while using $81\%$ less data.

## Prerequisites

- [uv](https://docs.astral.sh/uv/)

## Data

- Download [CoT entropy data](https://huggingface.co/datasets/LabARSS/MMLU-Pro-chain-of-thought-entropy) for MMLU to `data/out/cot_entropy`
- Download [reasoning data](https://huggingface.co/datasets/LabARSS/MMLU-Pro-reasoning-entropy-Qwen3-8B) for MMLU to `data/out/reasoning_entropy`

Other datasets are included in the repo and also published on Huggingface:
- [MMLU Pro education Level](https://huggingface.co/datasets/LabARSS/MMLU-Pro-education-level)
- [MMLU Pro reasoning score](https://huggingface.co/datasets/LabARSS/MMLU-Pro-reasoning-score)
- [MMLU Pro single token entropy](https://huggingface.co/datasets/LabARSS/MMLU-Pro-single-token-entropy)

## Training pipeline

1. Main training pipeline - `src/experiments/pipeline/pipeline` 
2. Alternative baseline - `src/experiments/pipeline/alternative_baseline` 
3. Full distillation baseline - `src/experiments/pipeline/full_distill`
3. Full SFT baseline - `src/experiments/pipeline/sft_baseline`
3. Curriculum SFT baseline - `src/experiments/pipeline/sft_curriculum_baseline`

## Running experiments

`uv run src/experiments/REPLACE_ME.py`
