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

#%%
def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath, sep='\s+', engine='python')
    data.columns = [col.strip('"') for col in data.columns]
    data.index = [idx.strip('"') for idx in data.index]
    data = data.apply(pd.to_numeric)

    y = data['V1'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data.drop(columns=['V1']).values)

    return X_scaled, y