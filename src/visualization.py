import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA


def plot_consensus_metrics(eval_df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    specs = [
        ("Silhouette", "Silhouette Coefficient (SC)", "SC"),
        ("DBI", "Davies-Bouldin Index (DBI)", "DBI"),
        ("CHI", "Calinski-Harabasz Index (CHI)", "CHI"),
    ]

    for ax, (col, title, ylabel) in zip(axes, specs):
        ax.plot(eval_df["K"], eval_df[col], marker="o")
        ax.set_title(title)
        ax.set_xlabel("Jumlah Cluster (K)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_mean_rank(evaluation_kept, best_k):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    d = evaluation_kept.sort_values("K")
    bars = ax.bar(d["K"], d["Mean_Rank"])
    for bar, k in zip(bars, d["K"]):
        if int(k) == int(best_k):
            bar.set_linewidth(3)
            bar.set_edgecolor("black")
    ax.set_title("Mean Rank Multi-Index Validation")
    ax.set_xlabel("Jumlah Cluster (K)")
    ax.set_ylabel("Mean Rank (lebih kecil lebih baik)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_pca(baseline, aligned, ensemble_labels, best_k, random_state=42):
    X = baseline.to_numpy(dtype=float)
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X)

    panels = [
        ("K-Means", aligned["KM"]),
        ("K-Prototypes — aligned", aligned["KP"]),
        ("Gower K-Medoids — aligned", aligned["GM"]),
        ("Ensemble Konsensus", ensemble_labels),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for ax, (title, labels) in zip(axes, panels):
        labels = np.asarray(labels)
        for cluster in np.sort(np.unique(labels)):
            idx = labels == cluster
            ax.scatter(
                X_pca[idx, 0], X_pca[idx, 1],
                s=16, alpha=0.65, label=f"C{cluster + 1}"
            )
        ax.set_title(f"{title} (K={best_k})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


def plot_cluster_distribution(labels, best_k):
    sizes = pd.Series(labels).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar([f"C{i+1}" for i in sizes.index], sizes.values)
    ax.set_title(f"Jumlah Anggota Tiap Cluster — Ensemble (K={best_k})")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Jumlah UMKM")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_cluster_heatmap(typeaware, labels, numerical_columns, categorical_columns):
    source = typeaware.copy()
    source["Cluster"] = np.asarray(labels) + 1
    features = numerical_columns + categorical_columns
    means = source.groupby("Cluster")[features].mean(numeric_only=True)

    fig, ax = plt.subplots(
        figsize=(max(10, 1.5 * len(features)), 5)
    )
    sns.heatmap(
        means, annot=True, fmt=".2f", ax=ax,
        cbar_kws={"label": "Mean"}
    )
    ax.set_title("Heatmap Rata-rata Fitur per Cluster")
    ax.set_xlabel("Fitur")
    ax.set_ylabel("Cluster")
    fig.tight_layout()
    return fig


def build_profile_tables(typeaware, labels, numerical_columns, categorical_columns):
    source = typeaware.copy()
    source["Cluster"] = np.asarray(labels) + 1
    total = len(source)

    numeric_rows = []
    for cluster_id, grp in source.groupby("Cluster"):
        row = {
            "Cluster": f"C{cluster_id}",
            "n": len(grp),
            "n (%)": round(100 * len(grp) / total, 1),
        }
        for col in numerical_columns:
            median = grp[col].median()
            q1 = grp[col].quantile(0.25)
            q3 = grp[col].quantile(0.75)
            row[f"{col} (median)"] = round(float(median), 2)
            row[f"{col} (IQR P25-P75)"] = f"{q1:,.2f} - {q3:,.2f}"
        numeric_rows.append(row)

    categorical_rows = []
    for cluster_id, grp in source.groupby("Cluster"):
        n = len(grp)

        nib = pd.to_numeric(grp["NIB/SKU"], errors="coerce")
        sosmed = pd.to_numeric(grp["Sosmed"], errors="coerce")
        marketplace = pd.to_numeric(grp["Marketplace"], errors="coerce")
        lahan = pd.to_numeric(grp["Kepemilikan Lahan"], errors="coerce")

        categorical_rows.append({
            "Cluster": f"C{cluster_id}",
            "n": n,
            "% NIB/SKU (Ada)": round(100 * nib.eq(1).mean(), 1),
            "% Memakai Sosmed": round(100 * sosmed.gt(0).mean(), 1),
            "% Memakai Marketplace": round(100 * marketplace.gt(0).mean(), 1),
            "% Lahan Milik Sendiri": round(100 * lahan.eq(1).mean(), 1),
        })

    return (
        pd.DataFrame(numeric_rows),
        pd.DataFrame(categorical_rows),
    )
