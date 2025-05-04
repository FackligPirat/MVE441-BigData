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
# %% Max first test

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_num)

models = {
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=0),
}

# Range of number of features to test
k_values = list(range(1, 257, 1))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

def evaluate_feature_count(clf):
    scores = []
    for k in k_values:
        pipe = Pipeline([
            ("select", SelectKBest(score_func=f_classif, k=k)),
            ("clf", clf)
        ])
        mean_cv_score = cross_val_score(pipe, X_scaled, y_num, cv=cv).mean()
        scores.append(mean_cv_score)
        if k % 25 == 0:
            print(f'Done with {k} of {len(k_values)}')
    best_k = k_values[np.argmax(scores)]
    best_score = max(scores)
    return best_k, best_score, scores

# Run CV-based feature selection
best_k_knn, score_knn, scores_knn = evaluate_feature_count(models["KNN (k=5)"])
best_k_rf, score_rf, scores_rf = evaluate_feature_count(models["Random Forest"])

# Plot results
plt.plot(k_values, scores_knn, label=f"KNN (best k={best_k_knn})")
plt.plot(k_values, scores_rf, label=f"Random Forest (best k={best_k_rf})")
plt.xlabel("Number of Features Selected")
plt.ylabel("Cross-Validated Accuracy")
plt.title("Optimal Number of Features via Filtering (F-test)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Report results
print(f"KNN: Best number of features = {best_k_knn}, Accuracy = {score_knn:.3f}")
print(f"Random Forest: Best number of features = {best_k_rf}, Accuracy = {score_rf:.3f}")

selector_knn = SelectKBest(score_func=f_classif, k=best_k_knn)
selector_knn.fit(X_scaled, y_num)
selected_knn = selector_knn.get_support(indices=True)  # indices of selected pixels

# Fit SelectKBest with best k for RF
selector_rf = SelectKBest(score_func=f_classif, k=best_k_rf)
selector_rf.fit(X_scaled, y_num)
selected_rf = selector_rf.get_support(indices=True)

# Display selected pixel indices
print(f"KNN selected pixel indices ({best_k_knn} features):\n{selected_knn}")
print(f"Random Forest selected pixel indices ({best_k_rf} features):\n{selected_rf}")

def plot_selected_pixels(selected_indices, title):
    mask = np.zeros(256, dtype=bool)
    mask[selected_indices] = True
    plt.imshow(mask.reshape(16, 16), cmap="Greys", interpolation="none")
    plt.title(title)
    plt.axis("off")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plot_selected_pixels(selected_knn, f"KNN Selected Pixels (k={best_k_knn})")
plt.subplot(1, 2, 2)
plot_selected_pixels(selected_rf, f"RF Selected Pixels (k={best_k_rf})")
plt.tight_layout()
plt.show()
#%% Second test
def evaluate_wrapper_range(clf, feature_range):
    scores = []
    all_selected = []
    for n_feats in feature_range:
        sfs = SequentialFeatureSelector(
            clf,
            n_features_to_select=n_feats,
            direction='forward',
            cv=cv,
            n_jobs=-1
        )
        sfs.fit(X_scaled, y_num)
        selected_indices = sfs.get_support(indices=True)
        score = cross_val_score(clf, X_scaled[:, selected_indices], y_num, cv=cv).mean()
        scores.append(score)
        all_selected.append(selected_indices)
        if n_feats % 5 == 0:
            print(f"Wrapper: Done with {n_feats} features")
    best_idx = np.argmax(scores)
    return feature_range[best_idx], scores[best_idx], scores, all_selected[best_idx]

# Define range of features to test
feature_range = list(range(1, 257, 1))  # test from 5 to 50 features

# Run for KNN
best_k_knn_wrap, score_knn_wrap, scores_knn_wrap, selected_knn_wrap = evaluate_wrapper_range(
    KNeighborsClassifier(n_neighbors=5), feature_range
)

# Run for Random Forest
best_k_rf_wrap, score_rf_wrap, scores_rf_wrap, selected_rf_wrap = evaluate_wrapper_range(
    RandomForestClassifier(n_estimators=100, random_state=0), feature_range
)

# Plot wrapper CV results
plt.plot(feature_range, scores_knn_wrap, label=f"KNN (best k={best_k_knn_wrap})")
plt.plot(feature_range, scores_rf_wrap, label=f"RF (best k={best_k_rf_wrap})")
plt.xlabel("Number of Features Selected (Wrapper)")
plt.ylabel("Cross-Validated Accuracy")
plt.title("Wrapper-based Feature Selection (Forward Selection)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Report results
print(f"[Wrapper CV] KNN: Best number of features = {best_k_knn_wrap}, Accuracy = {score_knn_wrap:.3f}")
print(f"[Wrapper CV] RF:  Best number of features = {best_k_rf_wrap}, Accuracy = {score_rf_wrap:.3f}")
#%%
# Plot side-by-side comparison
plt.figure(figsize=(10, 5))

# KNN comparison
plt.subplot(1, 2, 1)
plt.plot(k_values, scores_knn, label="Filtering (F-test)")
plt.plot(feature_range, scores_knn_wrap, label="Wrapper (Forward)")
plt.axvline(best_k_knn, color="gray", linestyle="--", label=f"Best Filter k={best_k_knn}")
plt.axvline(best_k_knn_wrap, color="black", linestyle="--", label=f"Best Wrapper k={best_k_knn_wrap}")
plt.title("KNN: Filtering vs Wrapper")
plt.xlabel("Number of Selected Features")
plt.ylabel("CV Accuracy")
plt.legend()
plt.grid(True)

# RF comparison
plt.subplot(1, 2, 2)
plt.plot(k_values, scores_rf, label="Filtering (F-test)")
plt.plot(feature_range, scores_rf_wrap, label="Wrapper (Forward)")
plt.axvline(best_k_rf, color="gray", linestyle="--", label=f"Best Filter k={best_k_rf}")
plt.axvline(best_k_rf_wrap, color="black", linestyle="--", label=f"Best Wrapper k={best_k_rf_wrap}")
plt.title("Random Forest: Filtering vs Wrapper")
plt.xlabel("Number of Selected Features")
plt.ylabel("CV Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("Summary of Best Results:")
print(f"KNN    - Filtering: k={best_k_knn:3}, Accuracy={score_knn:.3f}")
print(f"KNN    - Wrapper  : k={best_k_knn_wrap:3}, Accuracy={score_knn_wrap:.3f}")
print(f"Random - Filtering: k={best_k_rf:3}, Accuracy={score_rf:.3f}")
print(f"Random - Wrapper  : k={best_k_rf_wrap:3}, Accuracy={score_rf_wrap:.3f}")
