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

from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate

def run_classification_experiment(X, y, mode="cv_no_tuning", random_state=42):
    """
    Train and evaluate classifiers using different modes:
    - 'cv_no_tuning': Perform cross-validation without tuning (default parameters).
    - 'grid_search_tuning': Perform hyperparameter tuning using GridSearchCV.
    - 'double_cv_tuning': Perform nested cross-validation (double CV).
    """
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

    models = {
        'kNN': {
            'classifier': KNeighborsClassifier(),
            'params': {'n_neighbors': [1, 3, 5, 7, 9]}
        },
        'LogReg': {
            'classifier': LogisticRegression(max_iter=1000),
            'params': {'C': [0.01, 0.1, 1, 10, 100]}
        },
        'RandomForest': {
            'classifier': RandomForestClassifier(random_state=random_state),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 5, 15] #Kanske borde ändra parametrar här?
            }
        }
    }

    results = {}

    for name, model_info in models.items():
        model = model_info['classifier']

        if mode == "cv_no_tuning":
            # Cross-validation without tuning
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            cv_acc = np.mean(cv_scores)  # Average CV accuracy
            
            model.fit(X_train, y_train)
        
        elif mode == "grid_search_tuning":
            # GridSearchCV tuning
            grid_search = GridSearchCV(model, model_info['params'], cv=5, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_  # Get best model
            cv_acc = grid_search.best_score_  # Best CV accuracy

        elif mode == "double_cv_tuning":
            # Nested cross-validation (double CV)
            grid_search = GridSearchCV(model, model_info['params'], cv=5, scoring='accuracy')
            nested_cv_scores = cross_val_score(grid_search, X, y, cv=5, scoring='accuracy')
            cv_acc = np.mean(nested_cv_scores)  # Average nested CV accuracy
            
            model.fit(X_train, y_train)  # Train final model

        elif mode == "holdout_cv":
            # Hold-out + Cross-validation mode
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            cv_acc = np.mean(cv_scores)  # CV Accuracy
            
            model.fit(X_train, y_train)  # Train model on full training set
            
        else:
            raise ValueError("Invalid mode. Choose from 'cv_no_tuning', 'grid_search_tuning', 'double_cv_tuning', 'holdout_cv'.")

        # Evaluate model
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))

        results[name] = {
            'Training Accuracy': train_acc,
            'CV Accuracy': cv_acc,
            'Test Accuracy': test_acc,
            'Optimism (CV - Train)': train_acc - cv_acc
        }

        print(f"\nConfusion Matrix for {name} ({mode} mode):")
        plot_confusion_matrix_sklearn(y_test, model.predict(X_test))

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

    # Run experiments with different tuning modes
    results_cv_no_tuning = run_classification_experiment(X, y, mode="cv_no_tuning")
    results_holdout_cv = run_classification_experiment(X, y, mode="holdout_cv")
    results_double_cv_tuning = run_classification_experiment(X, y, mode="double_cv_tuning")

    # Convert results to DataFrames
    df_cv_no_tuning = pd.DataFrame(results_cv_no_tuning).T
    df_holdout_cv = pd.DataFrame(results_holdout_cv).T
    df_double_cv_tuning = pd.DataFrame(results_double_cv_tuning).T

    # Merge DataFrames to compare metrics
    df_comparison = pd.concat([
        df_cv_no_tuning[['Training Accuracy', 'CV Accuracy', 'Test Accuracy', 'Optimism (CV - Train)']],
        df_holdout_cv[['Training Accuracy', 'CV Accuracy', 'Test Accuracy', 'Optimism (CV - Train)']],
        df_double_cv_tuning[['Training Accuracy', 'CV Accuracy', 'Test Accuracy', 'Optimism (CV - Train)']]
    ], axis=1)

    # Rename columns for clarity
    df_comparison.columns = [
        "Train_CV_no_tuning", "CV_CV_no_tuning", "Test_CV_no_tuning", "Optimism_CV_no_tuning",
        "Train_Holdout+CV", "CV_Holdout+CV", "Test_Holdout+CV", "Optimism_Holdout+CV",
        "Train_Double-CV", "CV_Double-CV", "Test_Double-CV", "Optimism_Double-CV"
    ]

    print("\n=== Accuracy & Optimism Comparison ===")
    print(df_comparison)

    # Extract models for x-axis
    models = df_comparison.index
    x = np.arange(len(models))
    width = 0.2  # Bar width

    # Plot comparison for Training, CV, and Test Accuracy
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    # Training Accuracy
    axes[0].bar(x - width, df_comparison["Train_CV_no_tuning"], width, label="CV-no-tuning")
    axes[0].bar(x, df_comparison["Train_Holdout+CV"], width, label="Holdout+CV")
    axes[0].bar(x + width, df_comparison["Train_Double-CV"], width, label="Double-CV")
    axes[0].set_title("Training Accuracy Comparison")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=15)
    axes[0].legend()

    # CV Accuracy
    axes[1].bar(x - width, df_comparison["CV_CV_no_tuning"], width, label="CV-no-tuning")
    axes[1].bar(x, df_comparison["CV_Holdout+CV"], width, label="Holdout+CV")
    axes[1].bar(x + width, df_comparison["CV_Double-CV"], width, label="Double-CV")
    axes[1].set_title("Cross-Validation Accuracy Comparison")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=15)
    axes[1].legend()

    # Test Accuracy
    axes[2].bar(x - width, df_comparison["Test_CV_no_tuning"], width, label="CV-no-tuning")
    axes[2].bar(x, df_comparison["Test_Holdout+CV"], width, label="Holdout+CV")
    axes[2].bar(x + width, df_comparison["Test_Double-CV"], width, label="Double-CV")
    axes[2].set_title("Test Accuracy Comparison")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=15)
    axes[2].legend()

    # Optimism Comparison (Training - CV Accuracy)
    axes[3].bar(x - width, df_comparison["Optimism_CV_no_tuning"], width, label="CV-no-tuning")
    axes[3].bar(x, df_comparison["Optimism_Holdout+CV"], width, label="Holdout+CV")
    axes[3].bar(x + width, df_comparison["Optimism_Double-CV"], width, label="Double-CV")
    axes[3].set_title("Optimism (Overfitting) Comparison")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(models, rotation=15)
    axes[3].legend()

    plt.tight_layout()
    plt.show()

    # SVD Visualization
    plot_svd_projection(X, y)

if __name__ == "__main__":
    main()
