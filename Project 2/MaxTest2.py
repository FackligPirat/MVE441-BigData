#%% Import libraries and functions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import time
from sklearn.model_selection import train_test_split
#%% Cats and dogs

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath, sep='\s+', engine='python')
    df = df.apply(pd.to_numeric)
    X = df.values
    return X

def rotate_image(img_flat, shape, rotation=0):
    return np.rot90(img_flat.reshape(shape,shape), k=rotation)

# Load cat/dog image data
X_catdog = load_and_preprocess_data("catdogdata.txt")

y_catdog = np.zeros(X_catdog.shape[0], dtype=int)
y_catdog[99:] = 1

indices = np.random.choice(len(X_catdog), 6, replace=False)

plt.figure(figsize=(12, 6))
for i, idx in enumerate(indices):
    plt.subplot(2, 3, i+1)
    plt.imshow(rotate_image(X_catdog[idx], 64, 3), cmap="gray")
    plt.title("Cat" if y_catdog[idx] == 0 else "Dog")
    plt.axis("off")
plt.tight_layout()
plt.show()
# %% Numbers
X_numbers = load_and_preprocess_data("Numbers.txt")

y_num = X_numbers[:, 0].astype(int)
X_num = X_numbers[:, 1:]
indices = np.random.choice(len(X_num), 6, replace=False)

plt.figure(figsize=(12, 6))
for i, idx in enumerate(indices):
    plt.subplot(2, 3, i+1)
    plt.imshow(rotate_image(X_num[idx],16,0), cmap="gray")
    plt.title(f"Label: {y_num[idx]}")
    plt.axis("off")
plt.tight_layout()
plt.show()

#%%
# Set up models
models = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=0)
}

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_catdog)

# Train/test split (hold-out)
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y_catdog, test_size=0.3, stratify=y_catdog, random_state=1
)

# Try different numbers of selected features
feature_counts = range(1, 51, 5)  # Try 1 to 50 features in steps of 5
results = {}

for name, model in models.items():
    scores = []
    print(f"Running forward selection for: {name}")
    for k in feature_counts:
        print(f"Done with {k} of {len(feature_counts)}")
        selector = SequentialFeatureSelector(
            model, n_features_to_select=k, direction='forward',
        )
        selector.fit(X_train, y_train)
        X_train_sel = selector.transform(X_train)
        X_val_sel = selector.transform(X_val)
        model.fit(X_train_sel, y_train)
        val_score = model.score(X_val_sel, y_val)
        scores.append(val_score)
    results[name] = scores

# Plotting
plt.figure(figsize=(10, 6))
for name, scores in results.items():
    plt.plot(feature_counts, scores, marker='o', label=name)
plt.xlabel("Number of Selected Features")
plt.ylabel("Validation Accuracy")
plt.title("Forward Selection with Hold-Out Validation (Cat vs. Dog)")
plt.legend()
plt.grid(True)
plt.show()