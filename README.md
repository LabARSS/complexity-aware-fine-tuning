# Code for "Complexity-aware fine-tuning" paper

General purpose Large Language Models (LLMs) are frequently fine-tuned to improve performance in niche domains. Although fine-tuning is a standard practice, we still lack a deep understanding of how to aggregate data for better results. In this work, we show that the entropy-based output estimation provides a meaningful guideline for fine-tuning data preparation. Specifically, across two small open models ~3B$ we find that a single token answer entropy shows ROC AUC score of ~0.73 and allows us to split the training data into three complexity categories. Moreover, we discover that these categories require different tuning mechanisms. Leveraging these insights, we propose a novel blueprint for efficient fine-tuning that outperforms the standard approach (TODO vs TODO accuracy). We also provide an in-depth analysis of alternative complexity estimation techniques based on expert assessment via model-as-judge (MASJ), entropy aggregation, and reasoning metadata with ROC AUC scores of 0.57, TODO and TODO accordingly. Our findings facilitate immediate enhancements in fine-tuning performance. In addition, we path the way to further investigation and immersion of the numerical complexity analysis.

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

@misc{goncharov2025complexityawarefinetuning,
      title={Complexity-aware fine-tuning}, 
      author={Andrey Goncharov and Daniil Vyazhev and Petr Sychev and Edvard Khalafyan and Alexey Zaytsev},
      year={2025},
      eprint={2506.21220},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.21220}, 
}
