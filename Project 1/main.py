#%% import and functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix
import seaborn as sns

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath, sep='\s+', engine='python')
    data.columns = [col.strip('"') for col in data.columns]
    data.index = [idx.strip('"') for idx in data.index]
    data = data.apply(pd.to_numeric)

    y = data['V1'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data.drop(columns=['V1']).values)

    return X_scaled, y

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

def evaluate_models_tuning(models, X_train, X_test, y_train, y_test, title="Training, Cross-Validation, and Test Error by Model"):
    results = {}
    for name, model_info in models.items():
        print(f"\nTuning {name}...")
        grid = GridSearchCV(model_info["model"], model_info["params"], cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        train_acc = accuracy_score(y_train, best_model.predict(X_train))
        test_acc = accuracy_score(y_test, best_model.predict(X_test))
        cv_acc = grid.best_score_

        results[name] = {
            "Best Params": grid.best_params_,
            "Train Accuracy": train_acc,
            "CV Accuracy": cv_acc,
            "Test Accuracy": test_acc,
            "Train Error": 1 - train_acc,
            "CV Error": 1 - cv_acc,
            "Test Error": 1 - test_acc,
            "Optimism": (1-train_acc) - (1-cv_acc)
        }

        print(f"  Best Params: {grid.best_params_}")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  CV Accuracy: {cv_acc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")

    print("\nResults:")
    for name, result in results.items():
        print(f"{name}: Train Error = {result['Train Error']:.4f}, CV Error = {result['CV Error']:.4f}, Test Error = {result['Test Error']:.4f}, Optimism = {result['Optimism']:.4f}")

    plot_model_errors(results, title)
    return results

def plot_model_errors(results, title="Training, Cross-Validation, and Test Error by Model"):

    error_data = pd.DataFrame(results).T[["Train Error", "CV Error", "Test Error"]]
    
    error_data.plot(kind='bar', figsize=(10, 6))
    plt.ylabel("Error Rate")
    plt.title(title)
    plt.xticks(rotation=0)
    plt.legend(title="Error Type")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix_percent(y_true, y_pred, labels, title="Confusion Matrix (%)"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
    cm_percent = np.round(cm_percent, 1)  # Round for display

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title(title, fontsize=16)
    plt.show()

def scrambler(y, p):
    n_to_change = int(len(y) * p)  
    indices_to_change = np.random.choice(len(y), size=n_to_change, replace=False)
    for idx in indices_to_change:
        i = np.random.randint(-9, 9)
        while i == y[idx]:
            i = np.random.randint(-9, 9)
        y[idx] = i
    return(y)

X, y = load_and_preprocess_data("Numbers.txt")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=420)

#%% 1.1

models = {
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "KNN (k=20)": KNeighborsClassifier(n_neighbors=20),
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1),
    "Random Forest": RandomForestClassifier(n_estimators= 100, max_depth=None, min_samples_split=2)
}

cv_results = {}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    cv_results[name] = scores
    print(f"{name}: Mean Accuracy = {scores.mean():.4f}, Std = {scores.std():.4f}")

plt.figure(figsize=(10, 6))
plt.bar(cv_results.keys(), [np.mean(s) for s in cv_results.values()], color='steelblue')
plt.ylabel("Cross-Validation Accuracy")
plt.title("Model Comparison (No Tuning)")
plt.ylim(0.7, 1.0)
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# boxplot
#plt.figure(figsize=(10, 6))
#plt.boxplot(cv_results.keys(), labels=cv_results.keys())
#plt.ylabel("Cross-Validation Accuracy")
#plt.title("Model Comparison (No Tuning)")
#plt.grid(True)
#plt.show()

#%% 1.1 (Confusion matrices)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    labels = np.unique(y)  
    plot_confusion_matrix_percent(y_test, y_pred, labels, title=f"{name} - Confusion Matrix (%)")
#%% 1.2
models_for_tuning = {
    "KNN": {"model": KNeighborsClassifier(), "params": {"n_neighbors":[1,3,5,10,20]}},
    "LR": {"model": LogisticRegression(max_iter=1000), "params": {"C":[0.01,0.1,1,10,100]}},
    "Random Forest": {"model": RandomForestClassifier(), "params": {"n_estimators":[50,100,200], 
                                                                  #"max_depth":[None, 10, 20], 
                                                                  #"min_samples_split": [2,5,10], 
                                                                  #"min_samples_leaf":[1,3,5]
                                                                  }}
}

best_models = {}
accuracy_results = {}

for name, model in models_for_tuning.items():
    print(f"Tuning {name}...")
    grid = GridSearchCV(model["model"], model["params"], cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    best_models[name] = best_model

    cv_acc = grid.best_score_
    test_acc = accuracy_score(y_test, best_model.predict(X_test))

    accuracy_results[name] = {
        "CV Accuracy": cv_acc,
        "Test Accuracy": test_acc
    }

    print(f"  Best Params: {grid.best_params_}")
    print(f"  CV Accuracy: {cv_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")

acc_df = pd.DataFrame(accuracy_results).T[["CV Accuracy", "Test Accuracy"]]

# Plot
acc_df.plot(kind='bar', figsize=(10, 6))
plt.ylim(0.7, 1.0)
plt.ylabel("Accuracy")
plt.title("CV and Test Accuracy by Model (After Tuning)")
plt.xticks(rotation=0)
plt.legend(title="Evaluation Set")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# %% Confusion matrix best models
for name, model in best_models.items():
    y_pred = model.predict(X_test)
    plot_confusion_matrix_percent(y_test, y_pred, np.unique(y), title=f"{name} (Tuned) - Confusion Matrix (%)")
#%% 1.3
results = evaluate_models_tuning(models_for_tuning,X_train, X_test,y_train, y_test)

#%% Part 2 
p_values = [0.1,0,4,0.8]

X, y = load_and_preprocess_data("Numbers.txt")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=420)

results_p = []

for p in p_values:
        y_new = scrambler(y_train,p)
        result = evaluate_models_tuning(models_for_tuning,X_train, X_test, y_new, y_test)
        results_p.append(result)
    
# %%
