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
    "SVM baseline", "LogReg baseline",
    "SNPS SVM", "SNPS LogReg",
    "SNPS SVM imp", "SNPS LogReg imp",
    "SNPS ens", "SNPS ens imp",
    "SVM", "LogReg"
    ]

    markers = ["o", "s", "o", "s", "o", "s", "P", "P", "o", "s"]
    linestyles = ["--", "--", "-", "-", "-", "-", "-", "-", "--", "--"]

    for col, label, marker, ls in zip(acc_cols, labels, markers, linestyles):
        plt.plot(
            x_values,
            df_q[col].values,
            marker=marker,
            linestyle=ls,
            label=label,
            linewidth=1,        # thinner lines
            markersize=5,       # smaller points
            alpha=0.8           # slight transparency (VERY useful)
        )

    plt.xlabel("Q range")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Q range")

    plt.xticks(x_values)  # only actual integer values
    handles, legend_labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], legend_labels[::-1], bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(alpha=0.3)

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
    "SVM baseline", "LogReg baseline",
    "SNPS SVM", "SNPS LogReg",
    "SNPS SVM imp", "SNPS LogReg imp",
    "SNPS ens", "SNPS ens imp",
    "SVM", "LogReg"
    ]

    markers = ["o", "s", "o", "s", "o", "s", "P", "P", "o", "s"]
    linestyles = ["--", "--", "-", "-", "-", "-", "-", "-", "--", "--"]

    for col, label, marker, ls in zip(acc_cols, labels, markers, linestyles):
        plt.plot(
            x_values,
            df_t[col].values,
            marker=marker,
            linestyle=ls,
            label=label,
            linewidth=1,        # thinner lines
            markersize=5,       # smaller points
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
    handles, legend_labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], legend_labels[::-1], bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(alpha=0.3)

    plt.savefig("SNPS acc/accuracy_vs_train_size.png")
    plt.close()


#--------------------sparsity plot-----------

def plot_vs_sparsity(file_path):
    df = pd.read_csv(file_path, sep=",")
    df = df.drop(columns=["timestamp"], errors="ignore")

    # keep only balanced rows (positive == negative)
    df_s = df[df["positive"] == df["negative"]].copy()
    df_s = df_s.sort_values("sparsity")

    acc_cols = [col for col in df.columns if col.startswith("acc_")]

    os.makedirs("SNPS acc", exist_ok=True)

    plt.figure()
    x_values = df_s["sparsity"].values

    labels = [
    "SVM baseline", "LogReg baseline",
    "SNPS SVM", "SNPS LogReg",
    "SNPS SVM imp", "SNPS LogReg imp",
    "SNPS ens", "SNPS ens imp",
    "SVM", "LogReg"
    ]

    markers = ["o", "s", "o", "s", "o", "s", "P", "P", "o", "s"]
    linestyles = ["--", "--", "-", "-", "-", "-", "-", "-", "--", "--"]

    for col, label, marker, ls in zip(acc_cols, labels, markers, linestyles):
        plt.plot(
            x_values,
            df_s[col].values,
            marker=marker,
            linestyle=ls,
            linewidth=1,
            markersize=5,
            alpha=0.8,
            label=label
        )

    plt.xlabel("Sparsity")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Sparsity")

    plt.xticks(x_values)
    handles, legend_labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], legend_labels[::-1], bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(alpha=0.3)

    plt.savefig("SNPS acc/accuracy_vs_sparsity.png")
    plt.close()


#-----------------------plot positive------------

def plot_vs_positive(file_path):
    df = pd.read_csv(file_path, sep=",")
    df = df.drop(columns=["timestamp"], errors="ignore")

    # keep only rows with fixed sparsity (0.8)
    df_p = df.iloc[10:21].copy()
    df_p = df_p.sort_values("positive")

    acc_cols = [col for col in df.columns if col.startswith("acc_")]

    os.makedirs("SNPS acc", exist_ok=True)

    plt.figure()
    x_values = df_p["positive"].values

    labels = [
    "SVM baseline", "LogReg baseline",
    "SNPS SVM", "SNPS LogReg",
    "SNPS SVM imp", "SNPS LogReg imp",
    "SNPS ens", "SNPS ens imp",
    "SVM", "LogReg"
    ]

    markers = ["o", "s", "o", "s", "o", "s", "P", "P", "o", "s"]
    linestyles = ["--", "--", "-", "-", "-", "-", "-", "-", "--", "--"]

    for col, label, marker, ls in zip(acc_cols, labels, markers, linestyles):
        plt.plot(
            x_values,
            df_p[col].values,
            marker=marker,
            linestyle=ls,
            linewidth=1,
            markersize=5,
            alpha=0.8,
            label=label
        )

    plt.xlabel("Positive ratio")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Positive ratio")

    plt.xticks(x_values, rotation=30)
    handles, legend_labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], legend_labels[::-1], bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(alpha=0.3)

    plt.savefig("SNPS acc/accuracy_vs_positive.png")
    plt.close()


def plot_legend_only():
    os.makedirs("SNPS acc", exist_ok=True)

    labels = [
        "SVM baseline", "LogReg baseline",
        "SNPS SVM", "SNPS LogReg",
        "SNPS SVM imp", "SNPS LogReg imp",
        "SNPS ens", "SNPS ens imp",
        "SVM", "LogReg"
    ]

    markers = ["o", "s", "o", "s", "o", "s", "P", "P", "o", "s"]
    linestyles = ["--", "--", "-", "-", "-", "-", "-", "-", "--", "--"]

    # Crea linee finte solo per la legenda
    handles = []
    for label, marker, ls in zip(labels, markers, linestyles):
        h, = plt.plot([], [], marker=marker, linestyle=ls,
                      linewidth=1, markersize=5, label=label)
        handles.append(h)

    # Crea figura SOLO per la legenda
    fig = plt.figure(figsize=(10, 2))  # larga e bassa → perfetta per 5 colonne
    fig.legend(
        handles=handles,
        labels=labels,
        loc="center",
        ncol=5,   # <-- 5 colonne
        frameon=False
    )

    plt.axis('off')

    plt.savefig("SNPS acc/legend_only.png", bbox_inches='tight', pad_inches=0)
    plt.close()


def main(plot):

    if plot == 1:
        file_path = "accuracies.csv"
        plot_vs_q_range(file_path)
    elif plot == 2:
        file_path = "accuracies.csv"
        plot_vs_train_size(file_path)
    elif plot == 3:
        file_path = "accuracies_sparsity.csv"
        plot_vs_sparsity(file_path)
    elif plot == 4:
        file_path = "accuracies_sparsity.csv"
        plot_vs_positive(file_path)
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main(1)
    main(2)
    main(3)
    main(4)