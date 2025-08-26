# EACL/NACL Progress steps

This document outlines the current status of experiments and steps.  
Completed steps are marked with `[x]`, pending ones with `[ ]`, if task in process `[~]`.

---

## Daniil Vyazhev steps:
- [~] Reread our article to see what should be added/changed.
- [ ] SFT by cross entropy (Qwen 3B)  
- [ ] SFT by cross entropy (Phi-4-mini)  
- [ ] Accuracy for fine-tuning pipelines after 10 epochs for Qwen 3B
    - [ ] Alternative pipeline
    - [ ] Distillation
- [ ] Accuracy for fine-tuning pipelines after 10 epochs for Phi-4-mini  
    - [ ] Alternative pipeline
    - [ ] Distillation

- [ ] To be continued...
---

## Andrey Goncharov steps:

- [~] Reread our article to see what should be added/changed.
- [ ] Paraphrase CoT to measure whether paraphrasing adds weight to questions.
    - [ ] Use large LLM to paraphrase medium questions 2 time and hard questions 4 times     
    - [ ] Train SFT anf SFT curriculum on the paraphrased dataset for 20 epochs
    - [ ] Repeat previous experiments wiht entropy for 20 epochs instead of 10
        - [ ] Train SFT without split for 20 epochs
            - [ ] Qwen 3B
            - [ ] Phi4 mini 
        - [ ] Train SFT curriculum for 20 epochs
            - [ ] Qwen 3B
            - [ ] Phi4 mini
        - [ ] Train distilled CoT without split for 20 epochs
            - [ ] Qwen 3B
            - [ ] Phi4 mini
        - [ ] Train alternative pipeline for 20 epochs
            - [ ] Qwen 3B
            - [ ] Phi4 mini
        - [ ] Train our pipeline for 20 epochs
            - [ ] Qwen 3B
            - [ ] Phi4 mini

---

## Alexey Zaytsev steps:

- [ ] Reread our article to see what should be added/changed.
- [ ] Questions that our research can answer.
      
- [ ] To be continued...
