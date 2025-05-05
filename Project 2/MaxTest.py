#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import SequentialFeatureSelector, SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

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
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)

#%% Filter step: Select top-k features using F-test
k_filter = 200
filter_selector = SelectKBest(score_func=f_classif, k=k_filter)
X_filtered = filter_selector.fit_transform(X, y)
selected_filter_mask = filter_selector.get_support()  # shape: (n_features,)

#%% Define Models
models = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=0),
    #"Neural Network": MLPClassifier(hidden_layer_sizes=(30,15), max_iter=2000, early_stopping=True, n_iter_no_change=10, validation_fraction=0.1)
}

#%% Forward Selection with Early Stopping
feature_counts = list(range(1, k_filter + 1))
cv = StratifiedKFold(n_splits=5, shuffle=True)

patience = 3
min_delta = 0.001

results = {}
selected_masks = {}

for model_name, model in models.items():
    print(f"\nEvaluating {model_name}...")
    mean_scores = []
    best_score = 0
    no_improvement_count = 0

    for k in feature_counts:
        sfs = SequentialFeatureSelector(model, n_features_to_select=k, direction='forward', cv=cv, n_jobs=-1)
        X_selected = sfs.fit_transform(X_filtered, y)
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
                print(f"Early stopping at {k} features: no improvement in last {patience} steps.")
                break

    results[model_name] = (feature_counts[:len(mean_scores)], mean_scores)

    # Combine filter mask and SFS mask to map back to original pixels
    sfs_mask = best_sfs.get_support()  # shape: (k_filter,)
    combined_mask = np.zeros(X.shape[1], dtype=bool)
    combined_mask[selected_filter_mask] = sfs_mask
    selected_masks[model_name] = combined_mask

#%% Plot CV Accuracy vs. Number of Features
plt.figure(figsize=(12, 6))
for model_name, (k_vals, scores) in results.items():
    plt.plot(k_vals, scores, marker='o', label=model_name)
plt.xlabel("Number of Selected Features")
plt.ylabel("Mean CV Accuracy")
plt.title("Forward Selection with Filter Step and Early Stopping")
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