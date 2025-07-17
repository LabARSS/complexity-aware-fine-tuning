# Complexity-Aware Splits for Fine-Tuning

This repository provides precomputed train/validation/test splits of MMLU data stratified by various difficulty metrics: entropy, entropy with fallback, cross-entropy, MASJ rating, and chain-of-thought reasoning steps. All splits are organized under `data/data_splits/`.

---

## Folder Layout

```
complexity-aware-fine-tuning/
└── data/
    └── data_splits/
        ├── notebooks/                        # Jupyter notebooks for EDA & splitting
        │   ├── entropy_splitter.ipynb
        │   ├── masj_splitter.ipynb
        │   └── reasoning_splitter.ipynb
        │
        ├── entropy/
        │   └── splits/
        │       ├── train_df_easy.tsv
        │       ├── valid_df_easy.tsv
        │       ├── train_df_middle.tsv
        │       ├── valid_df_middle.tsv
        │       ├── train_df_hard.tsv
        │       ├── valid_df_hard.tsv
        │       ├── test_combined_entropy.tsv
        │       └── test_balanced_combined_entropy.tsv
        │
        ├── entropy_fallback/
        │   └── splits/                        # same pattern as `entropy`
        │       ├── train_df_easy.tsv
        │       └── ...
        │
        ├── cross_entropy/
        │   └── splits/                        # same pattern as `entropy`
        │       ├── train_df_easy.tsv
        │       └── ...
        │
        ├── masj/
        │   └── splits/
        │       ├── train_df_high_school_and_easier.tsv
        │       ├── valid_df_high_school_and_easier.tsv
        │       ├── train_df_undergraduate.tsv
        │       ├── valid_df_undergraduate.tsv
        │       ├── train_df_graduate_and_postgraduate.tsv
        │       ├── valid_df_graduate_and_postgraduate.tsv
        │       ├── train_df_random.tsv
        │       ├── valid_df_random.tsv
        │       ├── test_combined_masj.tsv
        │       └── test_balanced_combined_masj.tsv
        │
        └── reasoning/
            └── splits/
                ├── train_df_easy.tsv
                ├── valid_df_easy.tsv
                ├── train_df_middle.tsv
                ├── valid_df_middle.tsv
                ├── train_df_hard.tsv
                ├── valid_df_hard.tsv
                ├── test_combined_reasoning.tsv
                └── test_balanced_combined_reasoning.tsv
```

---



## Splitting Criteria

- **Entropy / Entropy with Fallback / Cross-Entropy**  
  - **easy**: bottom 25% quantile  
  - **middle**: 25–75% quantiles  
  - **hard**: top 25% quantile  

- **MASJ Rating**  
  - Categories: `high_school_and_easier`, `undergraduate`, `graduate_and_postgraduate`, plus a `random` sample for baseline.  

- **Reasoning Steps**  
  - `low` → **easy**  
  - `medium` → **middle**  
  - `high` → **hard**  

Thresholds and sample sizes can be adjusted in the respective notebooks or scripts.

---
