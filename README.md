# Code for "Complexity-aware fine-tuning" paper

General-purpose Large Language Models (LLMs) are frequently fine-tuned through supervised fine-tuning (SFT) to enhance performance in specific domains. Better results can be achieved by distilling the chain-of-thought of a larger model at the cost of numerous expensive calls and a much greater amount of data.
We propose a novel blueprint for efficient fine-tuning that uses reasoning only for complex data identified by entropy. Specifically, across two small open models ($\approx 3B$) we split the training data into complexity categories by a single token answer entropy (ROC AUC $0.73$), fine-tune large language models (LLMs) via SFT and distillation, and show that our pipeline significantly outperforms the standard SFT approach ($0.55$ vs $0.43$ average accuracy) and provides comparable with distillation performance while using $62\\%$ less data ($0.55$ average accuracy for both).

**Note:** This is an ongoing research. If you want to reproduce the results from the EMNLP 2025 version, check out [this tag](https://github.com/LabARSS/complexity-aware-fine-tuning/releases/tag/emnlp2025).

## Prerequisites

- [uv](https://docs.astral.sh/uv/)

## Data

- Download [CoT entropy data](https://huggingface.co/datasets/LabARSS/MMLU-Pro-chain-of-thought-entropy) for MMLU to `data/out/cot_entropy`
- Download [reasoning data](https://huggingface.co/datasets/LabARSS/MMLU-Pro-reasoning-entropy-Qwen3-8B) for MMLU to `data/out/reasoning_entropy`

Other datasets are included in the repo and also published on Huggingface:
- [MMLU Pro education Level](https://huggingface.co/datasets/LabARSS/MMLU-Pro-education-level)
- [MMLU Pro reasoning score](https://huggingface.co/datasets/LabARSS/MMLU-Pro-reasoning-score)
- [MMLU Pro single token entropy](https://huggingface.co/datasets/LabARSS/MMLU-Pro-single-token-entropy)

## Running experiments

`uv run src/experiments/REPLACE_ME.py`

## Cite

```
@misc{goncharov2025complexityawarefinetuning,
      title={Complexity-aware fine-tuning}, 
      author={Andrey Goncharov and Daniil Vyazhev and Petr Sychev and Edvard Khalafyan and Alexey Zaytsev},
      year={2025},
      eprint={2506.21220},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.21220}, 
}
```
