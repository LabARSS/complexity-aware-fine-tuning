from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def visualize_all(
    df, x, hue, model_name=None, x_label="Entropy", xticks=None, save_to: str | None = None, bins: int | str = "auto"
):
    save_to_path = Path(__file__).parent.joinpath("../../../artifacts").joinpath(save_to) if save_to else None
    if save_to_path:
        save_to_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 6))
    ax = sns.histplot(df, x=x, hue=hue, hue_order=[False, True], multiple="dodge", bins=bins)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Count")
    if xticks is not None:
        ax.set_xticks(xticks[0], xticks[1])
    if hue is not None:
        plt.legend(
            handles=ax.get_legend().legend_handles, labels=["Incorrect", "Correct"], title="Answer", loc="upper right"
        )
    if model_name is not None:
        plt.title(model_name)

    if save_to_path is not None:
        plt.savefig(f"{save_to_path.absolute()}_main.pdf")

    if hue is not None:
        plt.figure(figsize=(14, 6))
        ax = sns.histplot(df, x=x, hue=hue, hue_order=[False, True], multiple="fill", bins=bins)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Proportion")
        if xticks is not None:
            ax.set_xticks(xticks[0], xticks[1])
        plt.legend(handles=ax.get_legend().legend_handles, labels=["Incorrect", "Correct"], title="Answer")
        if model_name is not None:
            plt.title(model_name)

        if save_to_path is not None:
            plt.savefig(f"{save_to_path.absolute()}_proportion.pdf")
