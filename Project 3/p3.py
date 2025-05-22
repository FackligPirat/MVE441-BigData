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
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from kneed import KneeLocator

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



#X = PCA().fit_transform(X)
P = PCA()
P.fit_transform(X)
explained_var= P.explained_variance_

knee = KneeLocator(range(1, len(explained_var)+1), explained_var, curve='convex', direction='decreasing')
n_components = knee.elbow or 2  # fallback = 2
print(n_components)
X = PCA(n_components=n_components).fit_transform(X)

K = range(2, 198)

fits = []
score = []
for k in K:
    # train the model for current value of k on training data
    model = KMeans(n_clusters=k, n_init="auto").fit(X)
    
    # append the model to fits
    fits.append(model)
    
    # Append the silhouette score to scores
    score.append(silhouette_score(X, model.labels_, metric='euclidean'))


sns.lineplot(x = K, y = score)
plt.show()

# === 4. Använd KneeLocator för att hitta bästa K ===
kl = KneeLocator(K, score, curve="convex", direction="decreasing")
best_k = kl.elbow
print(f"Optimalt antal kluster enligt Elbow: {best_k}")

Z = linkage(X, 'ward') # Ward Distance

dendrogram(Z) #plotting the dendogram

plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data point')
plt.ylabel('Distance')
plt.show()

def match_labels(y_true, y_pred):
    labels = np.zeros_like(y_pred)
    for cluster in np.unique(y_pred):
        mask = (y_pred == cluster)
        labels[mask] = mode(y_true[mask], keepdims=True)[0]
    return labels

def kmeanfunc(k,X):
    kmeans = KMeans(n_clusters=k, n_init="auto").fit(X)
    #gm = GaussianMixture(n_components=2, random_state=0).fit(X)

    # Prediktera kluster för varje punkt
    y_kmeans = kmeans.predict(X)

    y_matched = match_labels(y, y_kmeans)
    acc = accuracy_score(y, y_matched)
    return(acc)

listacc  = []
for k in K:
    listacc.append(kmeanfunc(k,X))

acclist = []
for i in range(5):
    acclist.append(kmeanfunc(4,X))

sns.lineplot(x = K, y = listacc)
plt.show()


print(np.mean(acclist))

# gaussian mixture model
# k-mean
# hierarchial clustering

# elbow
# cv
kmeans = KMeans(n_clusters=4, n_init="auto").fit(X)
y_kmeans = kmeans.predict(X)

# === 4. Visualisering ===
plt.figure(figsize=(8, 6))
palette = sns.color_palette("bright", n_colors=4)
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y_kmeans, palette=palette, s=60, edgecolor='k', alpha=0.8)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, marker='X', label='Centroider')
plt.title("KMeans-klustring i PCA-rum")
plt.xlabel("PCA Komponent 1")
plt.ylabel("PCA Komponent 2")
plt.legend(title="Kluster")
plt.grid(True)
plt.tight_layout()
plt.show()

# === 5. Visualisering med färg = kluster, symbol = sann etikett ===
plt.figure(figsize=(8, 6))
palette = sns.color_palette("bright", n_colors=len(np.unique(y_kmeans)))
markers = ['o', 's', '^', 'D', 'v', '*', 'X', 'P']  # olika symboler

for true_label in np.unique(y):
    mask = y == true_label
    sns.scatterplot(
        x=X[mask, 0],
        y=X[mask, 1],
        hue=y_kmeans[mask],
        palette=palette,
        marker=markers[true_label % len(markers)],
        edgecolor='k',
        s=60,
        alpha=0.8,
        legend=False
    )

# === 6. Centroider ===
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='black', s=200, marker='X', label='Centroider')

plt.title("KMeans-klustring i PCA-rum\n(Färg: Kluster, Symbol: Sann etikett)")
plt.xlabel("PCA Komponent 1")
plt.ylabel("PCA Komponent 2")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

def plot_pca_components(pca, image_shape, n_components=5):
    plt.figure(figsize=(12, 3))
    for i in range(n_components):
        comp = pca.components_[i].reshape(image_shape)
        plt.subplot(1, n_components, i + 1)
        plt.imshow(rotate_image(comp,64,3), cmap='seismic')  # 'seismic' för positiva/negativa mönster
        plt.title(f"PC {i+1}")
        plt.axis('off')
    plt.suptitle("PCA-komponenter visualiserade som bilder")
    plt.tight_layout()
    plt.show()

plot_pca_components(P, image_shape=(64, 64), n_components=15)