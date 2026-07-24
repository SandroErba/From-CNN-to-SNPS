import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker


def plot_vs_q_range(file_path):
    df = pd.read_csv(file_path, sep=",")
    df = df.drop(columns=["timestamp"])

    df_q = df[df["train_size"] == 5000].copy()
    df_q = df_q.sort_values("q_range")
    acc_cols = [col for col in df.columns if col.startswith("acc_")]

    os.makedirs("SNPS acc", exist_ok=True)

    plt.figure()

    x_values = df_q["q_range"].values

    labels = [
    "SVM baseline", "Logreg baseline",
    "SNPS SVM", "SNPS Logreg",
    "SNPS SVM importance", "SNPS Logreg importance",
    "SNPS ens", "SNPS ens importance",
    "SVM", "Logreg"
    ]

    markers = ["o", "s", "o", "s", "o", "s", "*", "*", "o", "s"]
    linestyles = ["-", "-", "-", "-", "--", "--", "-", "--", "-", "-"]

    for col, label, marker, ls in zip(acc_cols, labels, markers, linestyles):
        plt.plot(
            x_values,
            df_q[col].values,
            marker=marker,
            linestyle=ls,
            label=label,
            linewidth=1,        # thinner lines
            markersize=3,       # smaller points
            alpha=0.8           # slight transparency (VERY useful)
        )

    plt.xlabel("Q range")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Q range")

    plt.xticks(x_values)  # only actual integer values
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid()

    plt.savefig("SNPS acc/accuracy_vs_q_range.png")
    plt.close()


#----------------------train plot--------------------

def plot_vs_train_size(file_path):
    df = pd.read_csv(file_path, sep=",")
    df = df.drop(columns=["timestamp"])

    df_t = df[df["q_range"] == 11].copy()
    df_t = df_t.sort_values("train_size")

    acc_cols = [col for col in df.columns if col.startswith("acc_")]

    os.makedirs("SNPS acc", exist_ok=True)

    plt.figure()

    x_values = df_t["train_size"].values

    labels = [
    "SVM baseline", "Logreg baseline",
    "SNPS SVM", "SNPS Logreg",
    "SNPS SVM importance", "SNPS Logreg importance",
    "SNPS ens", "SNPS ens importance",
    "SVM", "Logreg"
    ]

    markers = ["o", "s", "o", "s", "o", "s", "*", "*", "o", "s"]
    linestyles = ["-", "-", "-", "-", "--", "--", "-", "--", "-", "-"]

    for col, label, marker, ls in zip(acc_cols, labels, markers, linestyles):
        plt.plot(
            x_values,
            df_t[col].values,
            marker=marker,
            linestyle=ls,
            label=label,
            linewidth=1,        # thinner lines
            markersize=3,       # smaller points
            alpha=0.8           # slight transparency (VERY useful)
        )

    plt.xlabel("Train size (k)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Train size")

    ax = plt.gca()
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k")
    )

    plt.xticks(x_values)  # only actual integer values
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid()

    plt.savefig("SNPS acc/accuracy_vs_train_size.png")
    plt.close()


def main(plot):
    file_path = "accuracies.csv"

    if plot == 1:
        plot_vs_q_range(file_path)
    elif plot == 2:
        plot_vs_train_size(file_path)
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main(1)
    main(2)