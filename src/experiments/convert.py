import pandas as pd

df = pd.read_json("data/out/distillation/mmlu_synth_gptoss_a_t0_8.jsonl", lines=True)

# df["input"] = df["input"].apply(lambda x: {**x, "question_id": int(x["question_id"])})

df.to_parquet(
    "data/out/distillation/mmlu_synth_gptoss_a_t0_8.parquet", index=False
)
