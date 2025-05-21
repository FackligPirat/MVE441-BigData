#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample

from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import SequentialFeatureSelector, SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from kneed import KneeLocator
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

#%% Load Data Functions
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath, sep='\s+', engine='python')
    df = df.apply(pd.to_numeric)
    X = df.values
    return X

def rotate_image(img_flat, shape, rotation=0): #So the images is correct orientation
    return np.rot90(img_flat.reshape(shape, shape), k=rotation)

def create_balanced_subset(X, y, n_per_class):
    X_balanced = []
    y_balanced = []

    for label in np.unique(y):
        X_class = X[y == label]
        y_class = y[y == label]

        X_sample, y_sample = resample(X_class, y_class, n_samples=n_per_class, random_state=42)
        X_balanced.append(X_sample)
        y_balanced.append(y_sample)

    return np.vstack(X_balanced), np.hstack(y_balanced)

def create_unbalanced_subset(X, y, class_sample_dict):
    X_unbalanced = []
    y_unbalanced = []

    for label, n_samples in class_sample_dict.items():
        X_class = X[y == label]
        y_class = y[y == label]

        X_sample, y_sample = resample(X_class, y_class, n_samples=n_samples, random_state=42)
        X_unbalanced.append(X_sample)
        y_unbalanced.append(y_sample)

    return np.vstack(X_unbalanced), np.hstack(y_unbalanced)

def plot_distribution(y, title):
    labels, counts = np.unique(y, return_counts=True)
    
    plt.figure()
    plt.bar(range(len(labels)), counts, tick_label=labels)
    plt.title(title)
    plt.xlabel("Class Label")
    plt.ylabel("Number of Samples")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

def plot_with_legend(X_2d, labels, title, label_prefix="Label"):
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', s=15)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    # Create a legend with unique labels
    unique_labels = np.unique(labels)
    handles = [mpatches.Patch(color=scatter.cmap(scatter.norm(l)), label=f"{label_prefix} {l}") 
               for l in unique_labels]
    plt.legend(handles=handles, title=label_prefix, loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#%% Select cats or dogs

use_catdog = False  # Set to False to use Numbers.txt
scaler = StandardScaler()

if use_catdog:
    X = load_and_preprocess_data("catdogdata.txt")
    y = np.zeros(X.shape[0], dtype=int)
    y[99:] = 1
    image_shape = (64, 64)

    indices = np.random.choice(len(X), 6, replace=False)
    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(indices):
        plt.subplot(2, 3, i+1)
        plt.imshow(rotate_image(X[idx], 64, 3), cmap="gray")
        plt.title("Cat" if y[idx] == 0 else "Dog")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
else:
    X_all = load_and_preprocess_data("Numbers.txt")
    y = X_all[:, 0].astype(int)
    X = X_all[:, 1:]
    image_shape = (16, 16)
    indices = np.random.choice(len(X), 6, replace=False)
    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(indices):
        plt.subplot(2, 3, i+1)
        plt.imshow(rotate_image(X[idx],16,0), cmap="gray")
        plt.title(f"Label: {y[idx]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
    # Part 2

    selected_labels = [0, 1, 2, 6]

    # Create mask
    mask = np.isin(y, selected_labels)

    # Filter data and labels
    X_filtered = X[mask]
    y_filtered = y[mask]

    X_filtered_scaled = scaler.fit_transform(X_filtered)

    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X_filtered_scaled)

    plot_with_legend(X_2d, y_filtered, "Ground Truth", label_prefix="Class")

    labels, counts = np.unique(y_filtered, return_counts=True)
    print("Class distribution before subsampling:")
    for label, count in zip(labels, counts):
        print(f"Label {label}: {count} instances")

    X_bal, y_bal = create_balanced_subset(X_filtered, y_filtered, n_per_class=100)

    labels_bal, counts_bal = np.unique(y_bal, return_counts=True)
    print("\nClass distribution in balanced subset:")
    for label, count in zip(labels_bal, counts_bal):
        print(f"Label {label}: {count} instances")

    class_sizes = {0: 100, 1: 300, 2: 50, 6: 150}
    X_unbal, y_unbal = create_unbalanced_subset(X_filtered, y_filtered, class_sizes)

    labels_unbal, counts_unbal = np.unique(y_unbal, return_counts=True)
    print("\nClass distribution in unbalanced subset:")
    for label, count in zip(labels_unbal, counts_unbal):
        print(f"Label {label}: {count} instances")

    plot_distribution(y_filtered, "Original Class Distribution")
    plot_distribution(y_bal, "Balanced Subset Distribution")
    plot_distribution(y_unbal, "Unbalanced Subset Distribution")

#%% Select data
data_choice = "unbalanced" # normal, balanced or unbalanced
if data_choice == "balanced":
    X = X_bal
    y = y_bal
    print(f'Using balanced dataset')
elif data_choice == "unbalanced":
    X = X_unbal
    y = y_unbal
    print(f'Using unbalanced dataset')
else:
    print(f'Usiung regular dataset')

#%% PCA and find number of clusters
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit PCA to find explained variance
pca = PCA().fit(X_scaled)

# Plot explained variance to find "elbow"
plt.figure(figsize=(8, 4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid()
plt.title('Explained Variance vs. Number of PCA Components')
plt.show()

# Optionally use KneeLocator to find the elbow
kl = KneeLocator(
    range(1, len(pca.explained_variance_ratio_) + 1),
    np.cumsum(pca.explained_variance_ratio_),
    curve="convex",
    direction="increasing"
)
#optimal_components = kl.knee
optimal_components = 40
print(f"Optimal number of PCA components: {optimal_components}")
pca = PCA(n_components=optimal_components)
X_pca = pca.fit_transform(X_scaled)

silhouette_scores = []
inertias = []
K = range(2, 20)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_pca)
    labels = kmeans.labels_
    
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_pca, labels))

# Plot Silhouette Score
plt.figure()
plt.plot(K, silhouette_scores, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method")
plt.grid()
plt.show()

# Plot Elbow Method
plt.figure()
plt.plot(K, inertias, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia (Within-cluster SSE)")
plt.title("Elbow Method")
plt.grid()
plt.show()

kneedle = KneeLocator(K, inertias, curve="convex", direction="decreasing")
optimal_k = kneedle.knee
print(f"Optimal number of clusters (Elbow): {optimal_k}")
# %%  Clustering with optimal_k
models = {
    "KMeans": KMeans(n_clusters=optimal_k, random_state=42),
    "GMM": GaussianMixture(n_components=optimal_k, random_state=42),
    "Agglomerative": AgglomerativeClustering(n_clusters=optimal_k)
}

for name, model in models.items():
    if name == "GMM":
        model.fit(X_pca)
        labels = model.predict(X_pca)
    else:
        labels = model.fit_predict(X_pca)
    
    score = silhouette_score(X_pca, labels)
    print(f"{name} Silhouette Score: {score:.4f}")
#%% Plot dendogram
X_subset = X_scaled

# Compute the linkage matrix (same method as model)
linkage_matrix = linkage(X_subset, method='ward')

# Plot dendrogram
plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix, truncate_mode='lastp', p=8)
plt.title("Dendrogram (Hierarchical Clustering)")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

k = 2  # or your optimal_k
labels = fcluster(linkage_matrix, k, criterion='maxclust')


#%% PCA visualization

pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

# Function to plot 2D scatter with legend
def plot_with_legend(X_2d, labels, title, label_prefix="Label"):
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', s=15)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    # Create a legend with unique labels
    unique_labels = np.unique(labels)
    handles = [mpatches.Patch(color=scatter.cmap(scatter.norm(l)), label=f"{label_prefix} {l}") 
               for l in unique_labels]
    plt.legend(handles=handles, title=label_prefix, loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot ground truth
plot_with_legend(X_2d, y, "Ground Truth", label_prefix="Class")

# Plot each clustering model
for name, model in models.items():
    if name == "GMM":
        model.fit(X_pca)
        cluster_labels = model.predict(X_pca)
    else:
        cluster_labels = model.fit_predict(X_pca)

    plot_with_legend(X_2d, cluster_labels, f"{name} Clustering (k={optimal_k})", label_prefix="Cluster")
