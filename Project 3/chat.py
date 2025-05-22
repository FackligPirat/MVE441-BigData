#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

#%% Load Data Functions
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath, sep='\s+', engine='python')
    df = df.apply(pd.to_numeric)
    X = df.values
    return X

def rotate_image(img_flat, shape, rotation=0): #So the images is correct orientation
    return np.rot90(img_flat.reshape(shape, shape), k=rotation)

#%% Select Dataset
use_catdog = True  # Set to False to use Numbers.txt

if use_catdog:
    X = load_and_preprocess_data("catdogdata.txt")
    y = np.zeros(X.shape[0], dtype=int)
    y[99:] = 1
    image_shape = (64, 64)

    indices = np.random.choice(len(X), 6, replace=False)
else:
    X_all = load_and_preprocess_data("Numbers.txt")
    y = X_all[:, 0].astype(int)
    X = X_all[:, 1:]
    image_shape = (16, 16)
    indices = np.random.choice(len(X), 6, replace=False)

# === 2. Elbow-metoden (KMeans) ===
inertias = []
K_range = range(1, 10)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(6, 4))
sns.lineplot(x=list(K_range), y=inertias, marker='o')
plt.title('Elbow Method (KMeans)')
plt.xlabel('Antal kluster')
plt.ylabel('Inertia')
plt.show()

# === 3. Cross-validation via BIC (Gaussian Mixture Model) ===
bics = []
aics = []
for k in range(1, 8):
    gmm = GaussianMixture(n_components=k, random_state=0)
    gmm.fit(X)
    bics.append(gmm.bic(X))
    aics.append(gmm.aic(X))

plt.figure(figsize=(6, 4))
sns.lineplot(x=list(range(1, 10)), y=bics, label='BIC', marker='o')
sns.lineplot(x=list(range(1, 10)), y=aics, label='AIC', marker='o')
plt.title('Model Selection for GMM (CV-proxy)')
plt.xlabel('Antal komponenter')
plt.ylabel('BIC / AIC')
plt.legend()
plt.show()

# === 4. Klustring med bästa antal kluster ===
best_k = 4  # Baserat på Elbow/BIC
print(f"Valt antal kluster: {best_k}")

# KMeans
kmeans = KMeans(n_clusters=best_k, random_state=0)
labels_kmeans = kmeans.fit_predict(X)

# Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=best_k)
labels_agglo = agglo.fit_predict(X)

# GMM
gmm = GaussianMixture(n_components=best_k, random_state=0)
labels_gmm = gmm.fit_predict(X)

# === 5. Visualisering av resultat ===
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

axs[0].scatter(X[:, 0], X[:, 1], c=labels_kmeans, cmap='tab10')
axs[0].set_title('KMeans')

axs[1].scatter(X[:, 0], X[:, 1], c=labels_agglo, cmap='tab10')
axs[1].set_title('Agglomerativ klustring')

axs[2].scatter(X[:, 0], X[:, 1], c=labels_gmm, cmap='tab10')
axs[2].set_title('Gaussian Mixture Model')

plt.tight_layout()
plt.show()