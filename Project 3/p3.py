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

kmeans = KMeans(n_clusters=4, n_init="auto").fit(X)
#gm = GaussianMixture(n_components=2, random_state=0).fit(X)

# Prediktera kluster för varje punkt
y_kmeans = kmeans.predict(X)

K = range(2, 8)
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

Z = linkage(X, 'ward') # Ward Distance

dendrogram(Z) #plotting the dendogram

plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data point')
plt.ylabel('Distance')
plt.show()

# gaussian mixture model
# k-mean
# hierarchial clustering

# elbow
# cv