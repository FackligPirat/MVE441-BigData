#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


#%% Load Data Functions and plot functions
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath, sep='\s+', engine='python')
    df = df.apply(pd.to_numeric)
    X = df.values
    return X

def rotate_image(img_flat, shape, rotation=0): #So the images is correct orientation
    return np.rot90(img_flat.reshape(shape, shape), k=rotation)

def plot_masks(results, selected_masks):
    plt.figure(figsize=(15, 4))
    for i, (model_name, mask) in enumerate(selected_masks.items()):
        plt.subplot(1, len(selected_masks), i+1)
        plt.imshow(mask.reshape(image_shape), cmap='gray')
        plt.title(f"{model_name} (Acc: {results[i]:.3f})")
        plt.axis("off")
    plt.suptitle(f"Selected Pixels by Lasso ({image_shape[0]}x{image_shape[1]})")
    plt.tight_layout()
    plt.show()

def plot_overlay(use_catdog, X, y, selected_masks):
    if use_catdog:
        sample_indices = [0, 1, 101]
        dim = 64
        rotation = 3
    else:
        sample_indices = [0, 1, 2]
        dim = 16
        rotation = 0

    X_samples = X[sample_indices]
    y_samples = y[sample_indices]

    plt.figure(figsize=(12, 4 * len(sample_indices)))

    for model_idx, (model_name, mask) in enumerate(selected_masks.items()):
        for i, idx in enumerate(sample_indices):
            image = rotate_image(X[idx].reshape(image_shape),dim,rotation)
            mask_img = mask.reshape(image_shape)

            plt.subplot(len(sample_indices), len(selected_masks), model_idx + i * len(selected_masks) + 1)
            plt.imshow(image, cmap='gray')  # base image
            plt.imshow(mask_img, cmap='autumn', alpha=0.4)  # overlay selected pixels
            plt.title(f"{model_name} | Label: {y[idx]}")
            plt.axis("off")

    plt.suptitle("Overlay of Selected Pixels on Example Images", fontsize=16)
    plt.tight_layout()
    plt.show()

#%% Select Dataset
use_catdog = True  # Set to False to use Numbers.txt

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

#ratio = 0.5
#X_unused, X, y_unused, y = train_test_split(X, y, test_size=ratio)

#%% IMPORTANT FOR LASSO
scaler = StandardScaler()
X = scaler.fit_transform(X)

#%% Filter step: Select top-k features using F-test
if use_catdog:
    k_filter = 200
else:
    k_filter = 50

filter_selector = SelectKBest(score_func=f_classif, k=k_filter)
X_filtered = filter_selector.fit_transform(X, y)
selected_filter_mask = filter_selector.get_support()  # shape: (n_features,)

#%% Define Models NN and RN takes really long time. Probably lower K_filter or increase delta
models = {
    "KNN": KNeighborsClassifier(n_neighbors= 14 if use_catdog else 7),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=10),
    #"Neural Network": MLPClassifier(hidden_layer_sizes=(30,15), max_iter=2000, early_stopping=True, n_iter_no_change=10, validation_fraction=0.1)
}

#%% Investigate Lasso

#if use_catdog:
#    threshholds = np.flip(np.arange(0.005, 0.09, 0.001))
#else:
#    threshholds = np.flip(np.arange(0.09, 0.5, 0.01))

max_features = 20

patience = 5
min_delta = 0.001 

with_filter = False
cv = StratifiedKFold(n_splits=5, shuffle=True)

results = np.zeros((max_features,len(models),2))
selected_masks = {}

for j, (model_name, model) in enumerate(models.items()):
    mean_scores = []
    best_score = 0
    no_improvement_count = 0

    print(f"Evaluating for {model_name}")

    for i in range(max_features):

        if with_filter:
            lasso = LassoCV(cv=cv, max_iter=10000)

            clf = lasso.fit(X_filtered, y)
            importance = np.abs(clf.coef_)

            idx = importance.argsort()[-(i+1)]
            th = importance[idx]

            # Get the selected features from Lasso
            lasso_selector = SelectFromModel(clf, prefit=True, threshold=th)
            X_lasso_selected = lasso_selector.transform(X_filtered)
            lasso_mask = lasso_selector.get_support()

            # Combine filter mask and Lasso mask to map back to original pixels
            mask = np.zeros(X.shape[1], dtype=bool)
            mask[selected_filter_mask] = lasso_mask
        else:
            lasso = LassoCV(cv=cv, max_iter=10000)

            clf = lasso.fit(X, y)
            importance = np.abs(clf.coef_)

            idx = importance.argsort()[-(i+1)]
            th = importance[idx]

            # Get the selected features from Lasso
            lasso_selector = SelectFromModel(clf, prefit=True, threshold=th)
            X_lasso_selected = lasso_selector.transform(X)
            mask = lasso_selector.get_support()

        if i%10 == 0:
            print(f"{mask.sum()} selected features")

        scores = cross_val_score(model, X_lasso_selected, y, cv=cv)
        mean_score = scores.mean()
        results[i,j,:] = [mean_score, mask.sum()]
        selected_masks[model_name] = mask

        # Early stopping logic
        if mean_score > best_score + min_delta:
            best_score = mean_score
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print(f"Early stopping at {i+1} features: no improvement in last {patience} steps.")
                break


        #print(f"{model_name}: mean CV accuracy = {mean_score:.4f}")

#%% Plot
plt.figure(figsize=(12, 6))
for i, (model_name, model) in enumerate(models.items()):
    x_results = results[:,i,1]
    idx = np.nonzero(x_results)[0]
    if len(idx) < max_features:
        idx = idx[:-patience]
    y_results = results[:,i,0]
    plt.plot(x_results[idx], y_results[idx], marker='o', label=model_name)

plt.xlabel("Number of Selected Features")
plt.ylabel("Mean CV Accuracy")
plt.title("Lasso with Early Stopping")
plt.legend()
plt.grid(True)
plt.ylim(bottom = 0, top=1)
plt.tight_layout()
plt.show()

#%% Plot masks
plot_masks(results[-1,:,0], selected_masks)
plot_overlay(use_catdog, X, y, selected_masks)

# %% Run for several runs

max_features = 20

num_runs = 5

patience = 5
min_delta = 0.001 

with_filter = False
cv = StratifiedKFold(n_splits=5, shuffle=True)

all_results = np.zeros((max_features,len(models),2,num_runs))

for run in range(num_runs):
    print(f"\n========== Run {run+1} ==========")

    for j, (model_name, model) in enumerate(models.items()):
        mean_scores = []
        best_score = 0
        no_improvement_count = 0

        print(f"Evaluating for {model_name}")

        for i in range(max_features):

            if with_filter:
                lasso = LassoCV(cv=cv, max_iter=10000)

                clf = lasso.fit(X_filtered, y)
                importance = np.abs(clf.coef_)

                idx = importance.argsort()[-(i+1)]
                th = importance[idx]

                # Get the selected features from Lasso
                lasso_selector = SelectFromModel(clf, prefit=True, threshold=th)
                X_lasso_selected = lasso_selector.transform(X_filtered)
                lasso_mask = lasso_selector.get_support()

                # Combine filter mask and Lasso mask to map back to original pixels
                mask = np.zeros(X.shape[1], dtype=bool)
                mask[selected_filter_mask] = lasso_mask
            else:
                lasso = LassoCV(cv=cv, max_iter=10000)

                clf = lasso.fit(X, y)
                importance = np.abs(clf.coef_)

                idx = importance.argsort()[-(i+1)]
                th = importance[idx]

                # Get the selected features from Lasso
                lasso_selector = SelectFromModel(clf, prefit=True, threshold=th)
                X_lasso_selected = lasso_selector.transform(X)
                mask = lasso_selector.get_support()

            if i%5 == 0:
                print(f"{mask.sum()} selected features")

            scores = cross_val_score(model, X_lasso_selected, y, cv=cv)
            mean_score = scores.mean()
            all_results[i,j,:,run] = [mean_score, mask.sum()]

            # Early stopping logic
            if mean_score > best_score + min_delta:
                best_score = mean_score
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    print(f"Early stopping at {i+1} features: no improvement in last {patience} steps.")
                    break

#%% Plot for several runs
#plt.figure(figsize=(12, 6))
fixed_colors = {
    "KNN": "blue",
    "Logistic Regression": "orange",
    "Random Forest": "green"
}

for i, (model_name, model) in enumerate(models.items()):
    color = fixed_colors[model_name]

    plt.figure(figsize=(8, 5))

    # Plot individual runs
    min_len = 1000

    for k in range(num_runs):
        x_results = all_results[:,i,1,k]
        idx = np.nonzero(x_results)[0]
        if len(idx) < max_features:
            idx = idx[:-patience]
        y_results = all_results[:,i,0,k]
        plt.plot(x_results[idx], y_results[idx], color=color, alpha=0.4, marker='o', linewidth=1)

        if len(idx) < min_len:
            min_len = len(idx)

    #Plot average
    avg_results = np.average(all_results[:,i,1,:], axis=1)
    plt.plot(avg_results[:min_len], all_results[:min_len,i,0,0], color=color, marker='o',
             linewidth=2.5, label=f"{model_name} (mean)")

    plt.xlabel("Number of Selected Features")
    plt.ylabel("Mean CV Accuracy")
    plt.title(f"Forward Selection Results for {model_name}")
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom = 0, top=1)
    plt.tight_layout()
    plt.show()

# %%
