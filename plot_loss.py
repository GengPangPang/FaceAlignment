import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


LOG_PATH = "checkpoints/HRNet_loss_log.csv"
OUT_DIR = "Loss"
OUT_PATH = os.path.join(OUT_DIR, "loss200.png")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(LOG_PATH)

    df["epoch"] = pd.to_numeric(df["epoch"])
    df["avg_loss"] = pd.to_numeric(df["avg_loss"])

    # 手动缩放到 10^-3 单位
    df["avg_loss_1e3"] = df["avg_loss"] * 1e3

    sns.set_theme(style="white")

    plt.figure(figsize=(8, 5))
    ax = sns.lineplot(
        data=df,
        x="epoch",
        y="avg_loss_1e3",
        linewidth=3.5,
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Average Loss ($\times 10^{-3}$)")

    # 不要网格，不要标题
    ax.grid(False)
    ax.set_title("")

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=300)
    plt.close()

    print(f"Saved loss curve to: {OUT_PATH}")


if __name__ == "__main__":
    main()