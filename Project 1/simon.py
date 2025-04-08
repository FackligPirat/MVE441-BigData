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

#### for part 2
def scrambler(y, p):
    n_to_change = int(len(y) * p)  
    indices_to_change = np.random.choice(len(y), size=n_to_change, replace=False)
    for idx in indices_to_change:
        i = np.random.randint(-9, 9)
        while i == y[idx]:
            i = np.random.randint(-9, 9)
        y[idx] = i
    return(y)

# 1. Load and preprocess data

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath, sep='\s+', engine='python')
    data.columns = [col.strip('"') for col in data.columns]
    data.index = [idx.strip('"') for idx in data.index]
    data = data.apply(pd.to_numeric)

    y = data['V1'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data.drop(columns=['V1']).values)

    return X_scaled, y

def setup(link,comp):
    x, y = (load_and_preprocess_data(link))
    svd = TruncatedSVD(n_components=comp)
    X_svd = svd.fit_transform(x)
    return(X_svd,y)

x,y = setup("Numbers.txt",20)
#### 1

notune_knn = cross_val_score(KNeighborsClassifier(n_neighbors=100), x, y, cv=5,scoring='accuracy').mean()

notune_lr = (cross_val_score(LogisticRegression(max_iter=1000), x, y, cv=5, scoring='accuracy').mean())

notune_randf = (cross_val_score(RandomForestClassifier(), x, y, cv=5, scoring='accuracy').mean())

##### 2

tune_knn = GridSearchCV(KNeighborsClassifier(), 
                                   {
                                       "n_neighbors": [3, 5, 7, 9]
                                       }, cv=5, scoring='accuracy').fit(x,y).best_score_

tune_lr = GridSearchCV(LogisticRegression(max_iter=1000), 
                                   {
                                       "C": [0.01, 0.1, 1, 10, 100]
                                       }, cv=5, scoring='accuracy').fit(x,y).best_score_

tune_randf = GridSearchCV(RandomForestClassifier(), 
                                   {
                                       "n_estimators": [50, 100, 200]
                                       }, cv=5, scoring='accuracy').fit(x,y).best_score_

### 3
print(tune_knn-notune_knn)
print(tune_lr-notune_lr)
print(tune_randf-notune_randf)
