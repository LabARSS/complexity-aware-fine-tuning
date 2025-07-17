# Research direction

Date: Jul 14, 2025. The goal of this doc is to align the research team on the further research direction of the project. 

## Where we currently are

Current state: 

1. We briefly explored how fine-tuning small models on easy/medium/hard MMLU-Pro questions affects the accuracy with and without CoT (chain-of-thought).
2. We assembled a pipeline to train on easy and medium data with SFT and on the hard data with SFT on distilled CoT and achieved results comparable to fine-tuning on the whole dataset with distilled CoT. It is a good thing since we are saving data.
3. We compared ROC AUC of single-token answer entropy and some CoT entropy aggregates. We found that single-token answer entropy provides better ROC AUC score.

What is missing:

1. We measured accuracy on the combined test dataset. Combined means that we had questions from all complexity categories in the test split, e.g. we trained on easy data, but measured on the dataset with easy, medium and hard data. 
2. We measured accuracy using CoT when we trained on the distilled CoT and measured it without CoT when we did plain SFT. We do not know how plain SFT training affects CoT response accuracy and vice versa. 
3. We did not train long enough to reach a plateau. 
4. We did not utilize out knowledge of the correct answers in the complexity estimate.
5. We did not try semantically-meaningful CoT aggregates: there must be a ton of high-confidence/low-meaning tokens in the sequence, e.g. "a", "the".

## Where to go from here

1. Polish the current pipeline
   1. Repeat fine-tuning experiments for more epochs. 
      1. See the plateau.
   2. Perform SFT and distilled CoT training on different complexity bands measuring across: complexity bands separately, combined test split, with and without CoT in the response for both methods. 
      1. See how training on easy data affects results on hard questions.
      2. See how SFT training affects results with CoT in the response.
   3. Utilize knowledge of the correct answers
      1. Do the same experiments as above for the cross entropy as a complexity metric.
2. Build on top of the current pipeline
   1. Analyze much smaller data chunks. Do more granular splitting.
      1. In the light of [POLARIS](https://hkunlp.github.io/blog/2025/Polaris/), we need to see if we should be discarding easy questions instead of doing SFT on them.
   2. Analyze the shift in the complexity after one cycle of training. Apply pipeline recursively until we stop seeing the shift.
   3. Add another dataset.
   4. Replace training on the distilled CoT with RL.
   5. Add RL penalty on the length of the reasoning chain based on the question complexity.
3. Seek a better complexity metric
   1. Experiment with semantically-meaningful CoT aggregates: leave only nouns and verbs, discard "a", "the" and other high-confidence/low-signal tokens, etc.