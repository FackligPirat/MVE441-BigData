#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 1. Load and preprocess data

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath, sep='\s+', engine='python')
    data.columns = [col.strip('"') for col in data.columns]
    data.index = [idx.strip('"') for idx in data.index]
    data = data.apply(pd.to_numeric)

    y = data['V1'].values
    X = data.drop(columns=['V1']).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Data shape:", X.shape)
    print("Unique labels:", np.unique(y))

    return X_scaled, y


# 2. Visualize sample digits

def visualize_samples(X, y, n_samples=9):
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for ax in axes.flatten():
        idx = np.random.randint(0, X.shape[0])
        img = X[idx].reshape(16, 16)
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Label: {y[idx]}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# 3. Train and tune classifiers

def run_classification_experiment(X, y, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

    models = {
        'kNN': {
            'classifier': KNeighborsClassifier(),
            'params': {'n_neighbors': [1, 3, 5, 7, 9, 11, 15, 20]}
        },
        'LogReg': {
            'classifier': LogisticRegression(max_iter=1000),
            'params': {'C': [0.01, 0.1, 1, 10, 100]}
        },
        'RandomForest': {
            'classifier': RandomForestClassifier(random_state=random_state),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20]
            }
        }
    }

    results = {}

    for name, model_info in models.items():
        grid_search = GridSearchCV(model_info['classifier'], model_info['params'], cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        train_acc = accuracy_score(y_train, best_model.predict(X_train))
        cv_acc = grid_search.best_score_
        test_acc = accuracy_score(y_test, best_model.predict(X_test))

        results[name] = {
            'Best Params': grid_search.best_params_,
            'Training Accuracy': train_acc,
            'CV Accuracy': cv_acc,
            'Test Accuracy': test_acc,
            'Optimism (CV - Train)': train_acc - cv_acc
        }

        #Confusion Matrix for better performance insight
        print(f"\nConfusion Matrix for {name}:")
        plot_confusion_matrix_sklearn(y_test, y_pred)

    return results


# 4. SVD Visualization

def plot_svd_projection(X, y):
    svd = TruncatedSVD(n_components=100)
    X_svd = svd.fit_transform(X)

    plt.figure(figsize=(12, 4))

    #Plot 1: Explained variance
    plt.subplot(1, 2, 1)
    plt.plot(svd.explained_variance_, marker='o')
    plt.title("Explained Variance by SVD Components")
    plt.xlabel("Component")
    plt.ylabel("Explained Variance")

    #Plot 2: Projection on first two SVD components
    plt.subplot(1, 2, 2)
    plt.scatter(X_svd[:, 0], X_svd[:, 1], c=y, cmap='tab10', s=10)
    plt.title("Projection on First Two SVD Components")
    plt.xlabel("SVD 1")
    plt.ylabel("SVD 2")
    plt.tight_layout()
    plt.show()

# 4.5 confusion matrix
def plot_confusion_matrix_sklearn(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

# 5. Main

def main():
    X, y = load_and_preprocess_data("Numbers.txt")
    visualize_samples(X, y)
    results = run_classification_experiment(X, y)

    print("\n=== Tuned Model Performance Comparison ===")
    df = pd.DataFrame(results).T
    print(df)

    # Bar plot
    models = list(results.keys())
    x = np.arange(len(models))
    width = 0.2

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, [results[m]['Training Accuracy'] for m in models], width, label='Train')
    plt.bar(x, [results[m]['CV Accuracy'] for m in models], width, label='CV')
    plt.bar(x + width, [results[m]['Test Accuracy'] for m in models], width, label='Test')
    plt.ylabel("Accuracy")
    plt.title("Tuned Model Accuracy Comparison")
    plt.xticks(x, models)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # SVD Visualization
    plot_svd_projection(X, y)

if __name__ == "__main__":
    main()

# %%
