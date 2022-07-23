# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 17:52:28 2022

@author: Maria_Marousopoulou
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# itemFeatures = pd.read_csv('dressipi_recsys2022/mini_dataset.csv')
itemFeatures = pd.read_csv('dressipi_recsys2022/dataset.csv')
itemFeatures = itemFeatures.sort_values(by=['session_id'])
itemFeatures = itemFeatures.reset_index(drop=True)
finalItemFeatures = itemFeatures.fillna(0)

label = finalItemFeatures[['label']]
finalItemFeatures = finalItemFeatures.drop(columns=['label', 'date', 'session_id', 'item_id'])

X_train, X_test, y_train, y_test = train_test_split(finalItemFeatures, label, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

number_features = 73

# pca = PCA()
pca = PCA(n_components=number_features)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_
print(explained_variance)
total_explained_variance = np.sum(explained_variance)
percentage_total_explained_variance = total_explained_variance * 100
print('Number of features:', number_features)
print('Total explained variance:', total_explained_variance)
print('Percentage of Total explained variance:', percentage_total_explained_variance, '%')
