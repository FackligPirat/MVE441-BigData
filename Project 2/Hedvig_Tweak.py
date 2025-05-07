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

#%% IMPORTANT FOR LASSO
scaler = StandardScaler()
X = scaler.fit_transform(X)

#%% Filter step: Select top-k features using F-test
k_filter = 200 #Select the "best" 200 features
#Maybe use higher k_filter for cats and dogs and lower for num
#if use_catdog:
    #k_filter = 400

filter_selector = SelectKBest(score_func=f_classif, k=k_filter)
X_filtered = filter_selector.fit_transform(X, y)
selected_filter_mask = filter_selector.get_support()  # shape: (n_features,)

#%% Define Models NN and RN takes really long time. Probably lower K_filter or increase delta
models = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=0),
    #"Neural Network": MLPClassifier(hidden_layer_sizes=(30,15), max_iter=2000, early_stopping=True, n_iter_no_change=10, validation_fraction=0.1)
}

#%% Investigate Lasso

#if use_catdog:
#    threshholds = np.flip(np.arange(0.005, 0.09, 0.001))
#else:
#    threshholds = np.flip(np.arange(0.09, 0.5, 0.01))

max_features = 10

with_filter = False
cv = StratifiedKFold(n_splits=5, shuffle=True)

results = np.zeros((max_features,len(models),3))
selected_masks = {}

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

    print(f"Evaluating with {mask.sum()} selected features")

    for j, (model_name, model) in enumerate(models.items()):
        scores = cross_val_score(model, X_lasso_selected, y, cv=cv)
        mean_score = scores.mean()
        results[i,j,:] = [mean_score, mask.sum(), th]
        selected_masks[model_name] = mask

        #print(f"{model_name}: mean CV accuracy = {mean_score:.4f}")

#%% Plot
plt.figure(figsize=(12, 6))
for i, (model_name, model) in enumerate(models.items()):
    plt.plot(results[:,i,1], results[:,i,0], marker='o', label=model_name)
plt.xlabel("Number of Selected Features")
plt.ylabel("Mean CV Accuracy")
plt.title("Lasso with varying number of features (due to varying thresholds)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Plot masks
plot_masks(results[-1,:,0], selected_masks)
plot_overlay(use_catdog, X, y, selected_masks)

# %% 
