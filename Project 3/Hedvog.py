#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import resample

from sklearn.metrics import davies_bouldin_score

#%%
# === Load Data ===
def load_catdog_numbers():
    catdog_data = pd.read_csv("catdogdata.txt", sep='\s+', engine='python').values
    catdog_labels = np.zeros(catdog_data.shape[0], dtype=int)
    catdog_labels[99:] = 1

    numbers_data = pd.read_csv("Numbers.txt", sep='\s+', engine='python').values
    numbers_labels = numbers_data[:, 0].astype(int)
    numbers_data = numbers_data[:, 1:]

    return (catdog_data, catdog_labels), (numbers_data, numbers_labels)

# === Silhouette Plot ===
def plot_silhouette_samples(X, labels, title):
    sil_vals = silhouette_samples(X, labels)
    y_ticks = []
    y_lower, y_upper = 0, 0
    cluster_labels = np.unique(labels)
    for c in cluster_labels:
        c_sil_vals = sil_vals[labels == c]
        c_sil_vals.sort()
        y_upper += len(c_sil_vals)
        plt.barh(range(y_lower, y_upper), c_sil_vals, edgecolor='none')
        y_ticks.append((y_lower + y_upper) / 2)
        y_lower += len(c_sil_vals)
    plt.axvline(np.mean(sil_vals), color="k", linestyle="--")
    plt.yticks(y_ticks, [f"Cluster {c}" for c in cluster_labels])
    plt.xlabel("Silhouette Width")
    plt.ylabel("Observation")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# === Clustering Runner ===
def run_clustering_models(X, y_true=None, k_range=range(2, 11), dataset_name="Dataset", plot=True, diff_sample=False):
    X_scaled = StandardScaler().fit_transform(X)

    if dataset_name == "CatDog":
        n_components = min(25, X_scaled.shape[0], X_scaled.shape[1])
    else:
        if diff_sample:
            n_components = min(50, np.floor(X_scaled.shape[0]//5), X_scaled.shape[1])
            k_range = range(2, 21)
        else:
            n_components = min(50, X_scaled.shape[0], X_scaled.shape[1])

    X_pca_50 = PCA(n_components=n_components).fit_transform(X_scaled)
    X_pca_2 = PCA(n_components=2).fit_transform(X_scaled)

    results = []

    # The models
    models = [
        ('KMeans', lambda k: KMeans(n_clusters=k)),
        ('GMM', lambda k: GaussianMixture(n_components=k)),
        ('Agglomerative', lambda k: AgglomerativeClustering(n_clusters=k))
    ]

    for model_key, model_func in models:
        silhouettes, aris = [], []
        homogeneities, completenesses, v_measures = [], [], []
        labels_list = []
        davies_bouldin = []

        for k in k_range:
            model = model_func(k).fit(X_pca_50)
            labels = model.predict(X_pca_50) if model_key == 'GMM' else model.labels_
            labels_list.append(labels)
            silhouettes.append(silhouette_score(X_pca_50, labels))
            davies_bouldin.append(davies_bouldin_score(X_pca_50, labels))

            if y_true is not None:
                aris.append(adjusted_rand_score(y_true, labels))
                homogeneities.append(homogeneity_score(y_true, labels))
                completenesses.append(completeness_score(y_true, labels))
                v_measures.append(v_measure_score(y_true, labels))
            else:
                aris.append(np.nan)
                homogeneities.append(np.nan)
                completenesses.append(np.nan)
                v_measures.append(np.nan)

        if plot:
            # === SILHOUETTES ===
            # Plot metrics
            plt.figure(figsize=(10, 6))
            plt.plot(k_range, silhouettes, label='Silhouette', marker='o')
            #plt.plot(k_range, aris, label='ARI', marker='o')
            #plt.plot(k_range, homogeneities, label='Homogeneity', marker='o')
            #plt.plot(k_range, completenesses, label='Completeness', marker='o')
            #plt.plot(k_range, v_measures, label='V-Measure', marker='o')
            plt.title(f"{model_key} Metrics on {dataset_name}")
            plt.xlabel("Number of Clusters")
            plt.ylabel("Score")
            plt.ylim(bottom=0)
            plt.xticks(ticks=k_range)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

        # Best k
        best_k_sil = k_range[np.argmax(silhouettes)]
        best_labels_sil = labels_list[np.argmax(silhouettes)]
        if y_true is not None:
            best_k_ari = k_range[np.argmax(aris)]
            best_labels_ari = labels_list[np.argmax(aris)]
        else:
            best_k_ari = None
            best_labels_ari = None

        result = {
            'Dataset': dataset_name,
            'Model': model_key,
            'Best k (Sil)': best_k_sil,
            'Silhouette': max(silhouettes)
        }
        if y_true is not None:
            result.update({
                #'Best k (ARI)': best_k_ari,
                'ARI (Sil)': aris[np.argmax(silhouettes)],
                #'Homogeneity': homogeneity_score(y_true, best_labels_ari),
                #'Completeness': completeness_score(y_true, best_labels_ari),
                #'V-Measure': v_measure_score(y_true, best_labels_ari)
            })

        # PCA scatter and silhouette plot
        if plot:
            plt.figure(figsize=(6, 6))
            plt.scatter(X_pca_2[:, 0], X_pca_2[:, 1], c=best_labels_sil, cmap='viridis', s=40, alpha=0.7)
            plt.title(f"{model_key} Clustering (k={best_k_sil}, Silhouette) on {dataset_name}")
            plt.xlabel("PCA 1")
            plt.ylabel("PCA 2")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        # === DAVIES-BOULDIN ===
        # Plot metrics
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(k_range, davies_bouldin, marker='o')
            plt.title(f"{model_key} Metrics on {dataset_name}")
            plt.xlabel("Number of Clusters")
            plt.ylabel("Davies-Bouldin Index")
            #plt.ylim(bottom=0)
            plt.xticks(ticks=k_range)
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        # Best k
        best_k_db = k_range[np.argmin(davies_bouldin)]
        best_labels_db = labels_list[np.argmin(davies_bouldin)]

        result.update({
            'Best k (DB)': best_k_db,
            'Davies-Bouldin Index': min(davies_bouldin)
        })
        if y_true is not None:
            result.update({
                #'Best k (ARI)': best_k_ari,
                'ARI (DB)': aris[np.argmin(davies_bouldin)],
                #'Homogeneity': homogeneity_score(y_true, best_labels_ari),
                #'Completeness': completeness_score(y_true, best_labels_ari),
                #'V-Measure': v_measure_score(y_true, best_labels_ari)
            })

        results.append(result)

        # PCA scatter and silhouette plot
        if plot:
            plt.figure(figsize=(6, 6))
            plt.scatter(X_pca_2[:, 0], X_pca_2[:, 1], c=best_labels_db, cmap='viridis', s=40, alpha=0.7)
            plt.title(f"{model_key} Clustering (k={best_k_db}, Davies-Bouldin Index) on {dataset_name}")
            plt.xlabel("PCA 1")
            plt.ylabel("PCA 2")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        #plot_silhouette_samples(X_pca_50, best_labels_sil, f"{model_key} Silhouette Widths (k={best_k_sil})")

        #if best_k_ari is not None and best_k_ari != best_k_sil:
        #    plt.figure(figsize=(6, 5))
        #    plt.scatter(X_pca_2[:, 0], X_pca_2[:, 1], c=best_labels_ari, cmap='plasma', s=40, alpha=0.7)
        #    plt.title(f"{model_key} Clustering (k={best_k_ari}, ARI) on {dataset_name}")
        #    plt.xlabel("PCA 1")
        #    plt.ylabel("PCA 2")
        #    plt.grid(True)
        #    plt.tight_layout()
        #    plt.show()

    return results

# === Sample Size Simulation ===
def simulate_sample_size_effect(X, y, classes=[0,1,2], sizes=[200, 150, 100, 75, 50, 30, 20, 10], dataset_name="MNIST_SampleSize"):
    print(f"\n=== Simulating Sample Size Effect on {dataset_name} ===")
    all_results = []
    for n_per_class in sizes:
        X_sub, y_sub = [], []
        for cls in classes:
            cls_X = X[y == cls]
            X_sub.append(resample(cls_X, n_samples=n_per_class, random_state=42))
            y_sub.append(np.full(n_per_class, cls))
        X_combined = np.vstack(X_sub)
        y_combined = np.concatenate(y_sub)
        results = run_clustering_models(X_combined, y_true=y_combined, dataset_name=f"{dataset_name}_n={n_per_class*len(classes)}", plot=False, diff_sample=True)
        for r in results:
            r['Total Samples'] = len(y_combined)
        all_results.extend(results)
    return pd.DataFrame(all_results)

# === Imbalance Simulation ===
def simulate_imbalance_effect(X, y, imbalance_settings, dataset_name="MNIST_Imbalance"):
    print(f"\n=== Simulating Cluster Imbalance on {dataset_name} ===")
    all_results = []
    for imbalance in imbalance_settings:
        X_sub, y_sub = [], []
        for cls, n_samples in zip([0,1,2], imbalance):
            cls_X = X[y == cls]
            X_sub.append(resample(cls_X, n_samples=n_samples, random_state=42))
            y_sub.append(np.full(n_samples, cls))
        X_combined = np.vstack(X_sub)
        y_combined = np.concatenate(y_sub)
        results = run_clustering_models(X_combined, y_true=y_combined, dataset_name=f"{dataset_name}_ratio={imbalance}", plot=False, diff_sample=True)
        for r in results:
            r['Cluster Sizes'] = str(imbalance)
        all_results.extend(results)
    return pd.DataFrame(all_results)

# === Resampling-Based Stability ===
def resampling_stability_analysis(X, y_true, k, n_resamples=10, dataset_name="Dataset"):
    model_defs = {
        "KMeans": lambda: KMeans(n_clusters=k, random_state=42),
        "GMM": lambda: GaussianMixture(n_components=k, random_state=42),
        "Agglomerative": lambda: AgglomerativeClustering(n_clusters=k)
    }

    all_scores = []

    for model_name, model_func in model_defs.items():
        sil_scores, ari_scores = [], []

        for i in range(n_resamples):
            X_resampled, y_resampled = resample(X, y_true, random_state=42 + i)
            X_scaled = StandardScaler().fit_transform(X_resampled)
            n_components = min(50, X_scaled.shape[0], X_scaled.shape[1])
            X_pca = PCA(n_components=n_components).fit_transform(X_scaled)

            model = model_func()
            labels = model.fit_predict(X_pca) if model_name != "GMM" else model.fit(X_pca).predict(X_pca)

            sil_scores.append(silhouette_score(X_pca, labels))
            ari_scores.append(adjusted_rand_score(y_resampled, labels))

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.boxplot(sil_scores, vert=False)
        plt.title(f"{model_name} Silhouette Stability (k={k})")
        plt.xlabel("Silhouette Score")

        plt.subplot(1, 2, 2)
        plt.boxplot(ari_scores, vert=False)
        plt.title(f"{model_name} ARI Stability (k={k})")
        plt.xlabel("Adjusted Rand Index")

        plt.suptitle(f"{model_name} Stability on {dataset_name}")
        plt.tight_layout()
        plt.show()

        all_scores.append({
            'Model': model_name,
            'Silhouette Mean': np.mean(sil_scores),
            'Silhouette Std': np.std(sil_scores),
            'ARI Mean': np.mean(ari_scores),
            'ARI Std': np.std(ari_scores),
            'Dataset': dataset_name,
            'k': k,
            'n_resamples': n_resamples
        })

    return pd.DataFrame(all_scores)

#%%
# === Run ===
(catdog_X, catdog_y), (numbers_X, numbers_y) = load_catdog_numbers()
#results_catdog = run_clustering_models(catdog_X, y_true=catdog_y, dataset_name="CatDog", k_range=range(2, 11))
#results_numbers = run_clustering_models(numbers_X, y_true=numbers_y, dataset_name="Numbers", k_range=range(2, 16))

# Print results
#print("\n=== Full Dataset Results ===")
#print(pd.DataFrame(results_catdog + results_numbers))

# %% SAMPLE SIZE

mask_012 = np.isin(numbers_y, [0, 1, 2])
X_012 = numbers_X[mask_012]
y_012 = numbers_y[mask_012]

all_results = []

for _ in range(50):
    sample_size_results = simulate_sample_size_effect(X_012, y_012)
    all_results.append(sample_size_results)

# Concatenate all iteration results into a single DataFrame
combined_results = pd.concat(all_results, ignore_index=True)

# Compute average ARI and average k for each model and imbalance scenario
avg_results = combined_results.groupby(["Model", "Total Samples"]).agg({
    "ARI (Sil)": "mean",
    "Best k (Sil)": "mean"
}).reset_index()

sample_sizes = sample_size_results["Total Samples"].unique()
models = sample_size_results["Model"].unique()

#%%

for model in models:
    data = avg_results[avg_results["Model"] == model]
    ARI_sil = data["ARI (Sil)"].to_numpy()
    #ARI_db = data["ARI (DB)"].to_numpy()
    k_sil = data["Best k (Sil)"].to_numpy()
    #k_db = data["Best k (DB)"].to_numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, ARI_sil, label='Silhouette', marker='o')
    #plt.plot(sample_sizes, ARI_db, label='Davies-Bouldin', marker='o')
    plt.title(f"ARI for {model}")
    plt.xlabel("Sample Size")
    plt.ylabel("ARI Score")
    plt.ylim([0,0.72])
    plt.xlim(np.max(sample_sizes)+10, np.min(sample_sizes)-10)
    plt.xticks(ticks=sample_sizes)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, k_sil, label='Silhouette', marker='o')
    #plt.plot(sample_sizes, k_db, label='Davies-Bouldin', marker='o')
    plt.title(f"Number of Clusters for {model}")
    plt.xlabel("Sample Size")
    plt.ylabel("Number of Clusters")
    plt.ylim([0,21])
    plt.xlim(np.max(sample_sizes)+10, np.min(sample_sizes)-10)
    plt.xticks(ticks=sample_sizes)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

#%% BALANCE IMBALANCE
mask_012 = np.isin(numbers_y, [0, 1, 2])
X_012 = numbers_X[mask_012]
y_012 = numbers_y[mask_012]

imbalance_scenarios = [[100, 100, 100], [150, 100, 50], [200, 50, 50], [250, 25, 25]]

all_results = []

for _ in range(15):
    imbalance_results = simulate_imbalance_effect(X_012, y_012, imbalance_scenarios)
    all_results.append(imbalance_results)

# Concatenate all iteration results into a single DataFrame
combined_results = pd.concat(all_results, ignore_index=True)

# Compute average ARI and average k for each model and imbalance scenario
avg_results = combined_results.groupby(["Model", "Cluster Sizes"]).agg({
    "ARI (Sil)": "mean",
    "Best k (Sil)": "mean"
}).reset_index()

models = avg_results["Model"].unique()

for model in models:
    data = avg_results[avg_results["Model"] == model]
    ARI_sil = data["ARI (Sil)"].to_numpy()
    #ARI_db = data["ARI (DB)"].to_numpy()
    k_sil = data["Best k (Sil)"].to_numpy()
    #k_db = data["Best k (DB)"].to_numpy()

    x = range(len(imbalance_scenarios))

    plt.figure(figsize=(10, 6))
    plt.plot(x, ARI_sil, label='Silhouette', marker='o')
    plt.title(f"ARI for {model}")
    plt.xlabel("Imbalance Scenarios")
    plt.ylabel("ARI Score")
    plt.ylim([0,0.72])
    plt.xticks(ticks=x, labels=imbalance_scenarios)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(x, k_sil, label='Silhouette', marker='o')
    plt.title(f"Number of Clusters for {model}")
    plt.xlabel("Imbalance Scenarios")
    plt.ylabel("Number of Clusters")
    plt.ylim([0,14])
    plt.xticks(ticks=x, labels=imbalance_scenarios)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
# %%
