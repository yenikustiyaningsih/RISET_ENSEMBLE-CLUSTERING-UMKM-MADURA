import numpy as np
import pandas as pd
import kmedoids

from kmodes.kprototypes import KPrototypes
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.preprocessing import StandardScaler

NUMERICAL_COLUMNS = [
    "Jml. Tenaga Kerja",
    "Kapasitas Produksi/Thn",
    "Omset/Thn",
    "Aset",
]
CATEGORICAL_COLUMNS = [
    "NIB/SKU",
    "Sosmed",
    "Marketplace",
    "Kepemilikan Lahan",
]
FEATURE_ORDER = [
    "NIB/SKU",
    "Jml. Tenaga Kerja",
    "Kapasitas Produksi/Thn",
    "Omset/Thn",
    "Aset",
    "Sosmed",
    "Marketplace",
    "Kepemilikan Lahan",
]
GAMMA_KPROTO = 0.150004
N_INIT_GOWER = 10


def _sizes(labels):
    s = pd.Series(labels).value_counts()
    return int(s.min()), int(s.max())


def run_base_clustering(
    baseline, typeaware, gower_matrix, X_eval_combined,
    k_values, random_state=42
):
    X_kmeans = baseline[FEATURE_ORDER].to_numpy(dtype=float)

    kmeans_labels = {}
    kproto_labels = {}
    kmedoids_labels = {}

    eval_kmeans = []
    eval_kproto = []
    eval_kmedoids = []

    scaler = StandardScaler()
    data_kproto = typeaware.copy()
    data_kproto[NUMERICAL_COLUMNS] = scaler.fit_transform(
        data_kproto[NUMERICAL_COLUMNS]
    )
    X_kproto = data_kproto[
        NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS
    ].to_numpy()

    categorical_index = [
        X_kproto.shape[1] - len(CATEGORICAL_COLUMNS) + i
        for i in range(len(CATEGORICAL_COLUMNS))
    ]

    for k in k_values:
        # K-Means
        km = KMeans(
            n_clusters=k,
            init="k-means++",
            random_state=random_state,
            n_init=10,
            max_iter=300,
        )
        labels_km = km.fit_predict(X_kmeans)
        kmeans_labels[k] = labels_km
        mn, mx = _sizes(labels_km)
        eval_kmeans.append({
            "K": k,
            "Inertia": km.inertia_,
            "Silhouette": silhouette_score(X_kmeans, labels_km),
            "DBI": davies_bouldin_score(X_kmeans, labels_km),
            "CHI": calinski_harabasz_score(X_kmeans, labels_km),
            "Min_Cluster_Size": mn,
            "Max_Cluster_Size": mx,
        })

        # K-Prototypes
        kp = KPrototypes(
            n_clusters=k,
            init="Huang",
            n_init=10,
            max_iter=100,
            gamma=GAMMA_KPROTO,
            random_state=random_state,
        )
        labels_kp = kp.fit_predict(X_kproto, categorical=categorical_index)
        kproto_labels[k] = labels_kp
        mn, mx = _sizes(labels_kp)
        eval_kproto.append({
            "K": k,
            "Cost": kp.cost_,
            "Silhouette": silhouette_score(gower_matrix, labels_kp, metric="precomputed"),
            "DBI": davies_bouldin_score(X_eval_combined, labels_kp),
            "CHI": calinski_harabasz_score(X_eval_combined, labels_kp),
            "Min_Cluster_Size": mn,
            "Max_Cluster_Size": mx,
        })

        # Gower K-Medoids
        best_model = None
        for init_i in range(N_INIT_GOWER):
            candidate = kmedoids.fasterpam(
                gower_matrix,
                k,
                max_iter=100,
                random_state=random_state + init_i,
            )
            if best_model is None or candidate.loss < best_model.loss:
                best_model = candidate

        labels_gm = np.asarray(best_model.labels)
        kmedoids_labels[k] = labels_gm
        mn, mx = _sizes(labels_gm)
        eval_kmedoids.append({
            "K": k,
            "Loss": best_model.loss,
            "Silhouette": silhouette_score(gower_matrix, labels_gm, metric="precomputed"),
            "DBI": davies_bouldin_score(X_eval_combined, labels_gm),
            "CHI": calinski_harabasz_score(X_eval_combined, labels_gm),
            "Min_Cluster_Size": mn,
            "Max_Cluster_Size": mx,
        })

    return {
        "labels": {
            "kmeans": kmeans_labels,
            "kproto": kproto_labels,
            "kmedoids": kmedoids_labels,
        },
        "evaluation": {
            "kmeans": pd.DataFrame(eval_kmeans),
            "kproto": pd.DataFrame(eval_kproto),
            "kmedoids": pd.DataFrame(eval_kmedoids),
        },
    }
