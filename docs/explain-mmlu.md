1) **Main point**

Obtain a synthetic dataset (answers + brief explanations + analysis of erroneous answers + CoT tokens) for training subsequent models.

* original dataset: ```data\source\mmlu_pro_stem.tsv```

2) **Branches**

A: q + options $\rightarrow$ JSON{answer, rationale, key_steps(CoT)} $\rightarrow$ сравнение с gold.

B: q + options + gold $\rightarrow$ JSON{correct_answer, why_correct(CoT), distractor_analysis[ALL]}.

C: (1) q + options $\rightarrow$ JSON{answer,…}; (2) + gold $\rightarrow$ JSON{model_answer, is_correct, error_analysis, distractor_analysis[ALL]}.

* JSON can be replaced with tsv.

* distractor_analysis[ALL] explains why the other answers are incorrect