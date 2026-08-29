import numpy as np
import pandas as pd

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    confusion_matrix,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


def hungarian_alignment(reference_labels, target_labels):
    cm = confusion_matrix(reference_labels, target_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    return np.array([mapping[label] for label in target_labels])


def align_all_labels(labels, k_values):
    result = {}
    for k in k_values:
        km = np.asarray(labels["kmeans"][k])
        kp = np.asarray(labels["kproto"][k])
        gm = np.asarray(labels["kmedoids"][k])
        result[k] = {
            "KM": km,
            "KP": hungarian_alignment(km, kp),
            "GM": hungarian_alignment(km, gm),
        }
    return result


def majority_vote_with_tiebreak(km_labels, kp_labels, gm_labels):
    km_labels = np.asarray(km_labels)
    kp_labels = np.asarray(kp_labels)
    gm_labels = np.asarray(gm_labels)

    votes = np.column_stack([km_labels, kp_labels, gm_labels])
    consensus = np.empty(len(km_labels), dtype=int)

    for i in range(len(km_labels)):
        values, counts = np.unique(votes[i], return_counts=True)
        winners = values[counts == counts.max()]
        consensus[i] = winners[0] if len(winners) == 1 else km_labels[i]

    return consensus


def build_consensus(aligned, k_values):
    return {
        k: majority_vote_with_tiebreak(
            aligned[k]["KM"], aligned[k]["KP"], aligned[k]["GM"]
        )
        for k in k_values
    }


def evaluate_consensus(consensus_labels, gower_matrix, X_eval, k_values):
    rows = []
    for k in k_values:
        labels = consensus_labels[k]
        sizes = pd.Series(labels).value_counts()
        rows.append({
            "K": k,
            "Silhouette": silhouette_score(
                gower_matrix, labels, metric="precomputed"
            ),
            "DBI": davies_bouldin_score(X_eval, labels),
            "CHI": calinski_harabasz_score(X_eval, labels),
            "Min_Cluster_Size": int(sizes.min()),
            "Max_Cluster_Size": int(sizes.max()),
        })
    return pd.DataFrame(rows)


def rank_multi_index(evaluation, total_n):
    threshold = max(20, int(np.ceil(0.01 * total_n)))

    kept = evaluation[
        evaluation["Min_Cluster_Size"] >= threshold
    ].copy()
    dropped = evaluation[
        evaluation["Min_Cluster_Size"] < threshold
    ].copy()

    if kept.empty:
        raise ValueError(
            "Tidak ada kandidat K yang lolos filter ukuran cluster minimum."
        )

    kept["Rank_SC"] = kept["Silhouette"].rank(ascending=False, method="average")
    kept["Rank_DBI"] = kept["DBI"].rank(ascending=True, method="average")
    kept["Rank_CHI"] = kept["CHI"].rank(ascending=False, method="average")
    kept["Total_Rank"] = (
        kept["Rank_SC"] + kept["Rank_DBI"] + kept["Rank_CHI"]
    )
    kept["Mean_Rank"] = kept["Total_Rank"] / 3
    kept = kept.sort_values(["Mean_Rank", "K"]).reset_index(drop=True)

    return kept, dropped, threshold


def final_comparison(base_labels, ensemble_labels, best_k, gower_matrix, X_eval):
    rows = []

    for method_key, method_name in [
        ("kmeans", "K-Means"),
        ("kproto", "K-Prototypes"),
        ("kmedoids", "Gower K-Medoids"),
    ]:
        labels = np.asarray(base_labels[method_key][best_k])
        sizes = pd.Series(labels).value_counts()
        rows.append({
            "Metode": method_name,
            "Silhouette": silhouette_score(
                gower_matrix, labels, metric="precomputed"
            ),
            "DBI": davies_bouldin_score(X_eval, labels),
            "CHI": calinski_harabasz_score(X_eval, labels),
            "Min_Cluster_Size": int(sizes.min()),
            "Max_Cluster_Size": int(sizes.max()),
        })

    sizes = pd.Series(ensemble_labels).value_counts()
    rows.append({
        "Metode": "Ensemble Konsensus",
        "Silhouette": silhouette_score(
            gower_matrix, ensemble_labels, metric="precomputed"
        ),
        "DBI": davies_bouldin_score(X_eval, ensemble_labels),
        "CHI": calinski_harabasz_score(X_eval, ensemble_labels),
        "Min_Cluster_Size": int(sizes.min()),
        "Max_Cluster_Size": int(sizes.max()),
    })

    out = pd.DataFrame(rows)
    out["Rank_SC"] = out["Silhouette"].rank(ascending=False)
    out["Rank_DBI"] = out["DBI"].rank(ascending=True)
    out["Rank_CHI"] = out["CHI"].rank(ascending=False)
    out["Total_Rank"] = out[["Rank_SC", "Rank_DBI", "Rank_CHI"]].sum(axis=1)
    out["Mean_Rank"] = out["Total_Rank"] / 3
    return out.sort_values(["Mean_Rank", "Metode"]).reset_index(drop=True)
