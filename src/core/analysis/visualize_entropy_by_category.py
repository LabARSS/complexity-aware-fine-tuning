import matplotlib.pyplot as plt
import seaborn as sns


def visualize_entropy_by_category(df, category, x, hue, model_name=None):
    for cat in df[category].unique():
        plt.figure(figsize=(14, 6))
        ax = sns.histplot(
            df.loc[df[category].isin([cat]), :],
            x=x,
            hue=hue,
            hue_order=[False, True],
            multiple="dodge",
        )
        ax.set_xlabel("Entropy")
        ax.set_ylabel("Count")
        ax.set_title(cat if model_name is None else f"{cat} ({model_name})")
        plt.legend(handles=ax.get_legend().legend_handles, labels=["Incorrect", "Correct"], title="Answer")
