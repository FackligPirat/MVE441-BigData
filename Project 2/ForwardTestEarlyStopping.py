#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

#%% Load Data Functions
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath, sep='\s+', engine='python')
    df = df.apply(pd.to_numeric)
    X = df.values
    return X

def rotate_image(img_flat, shape, rotation=0):
    return np.rot90(img_flat.reshape(shape, shape), k=rotation)

#%% Select Dataset
use_catdog = False  # Set to False to use Numbers.txt

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

#%% Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#%% Define Models
models = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=0)
}

#%% Forward Selection with Early Stopping
feature_counts = range(1, X.shape[1] + 1, 1)
cv = StratifiedKFold(n_splits=5, shuffle=True)

patience = 2
min_delta = 0.05

results = {}
selected_masks = {}

for model_name, model in models.items():
    print(f"\nEvaluating {model_name}...")
    mean_scores = []
    best_score = 0
    no_improvement_count = 0

    for k in feature_counts:
        sfs = SequentialFeatureSelector(model, n_features_to_select=k, cv = cv, direction='forward', n_jobs=-1)
        X_selected = sfs.fit_transform(X_scaled, y)
        scores = cross_val_score(model, X_selected, y, cv=cv)
        mean_score = scores.mean()
        mean_scores.append(mean_score)
        print(f"{k} features: mean CV accuracy = {mean_score:.4f}")

        # Early stopping logic
        if mean_score > best_score + min_delta:
            best_score = mean_score
            no_improvement_count = 0
            best_sfs = sfs  # Save best selector
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print(f"Early stopping at {k} features: small improvment in last {patience} steps.")
                break

    results[model_name] = (feature_counts[:len(mean_scores)], mean_scores)
    selected_masks[model_name] = best_sfs.get_support()

#%% Plot CV Accuracy vs. Number of Features
plt.figure(figsize=(12, 6))
for model_name, (k_vals, scores) in results.items():
    plt.plot(k_vals, scores, marker='o', label=model_name)
plt.xlabel("Number of Selected Features")
plt.ylabel("Mean CV Accuracy")
plt.title("Forward Selection with Early Stopping")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Plot Selected Pixel Masks
plt.figure(figsize=(15, 4))
for i, (model_name, mask) in enumerate(selected_masks.items()):
    plt.subplot(1, len(selected_masks), i+1)
    plt.imshow(mask.reshape(image_shape), cmap='gray')
    plt.title(f"{model_name}")
    plt.axis("off")
plt.suptitle(f"Selected Pixels ({image_shape[0]}×{image_shape[1]})")
plt.tight_layout()
plt.show()