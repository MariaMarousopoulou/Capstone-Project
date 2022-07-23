# -*- coding: utf-8 -*-
"""
Created on Sat May 14 14:09:20 2022

@author: Maria_Marousopoulou
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# Create labels
trainPurchases = pd.read_csv('dressipi_recsys2022/train_purchases.csv')
trainSessions = pd.read_csv('dressipi_recsys2022/train_sessions.csv')

labels = trainPurchases.copy()
labels['label'] = 1

trainSessionsLabels = trainSessions.copy()
trainSessionsLabels['label'] = 0

labels = labels.append(trainSessionsLabels, ignore_index = True)
labels.sort_values(by=['session_id', 'date'], inplace = True)
labels.to_csv('dressipi_recsys2022/labels.csv', index = False)


# Split into train / test
x_data = labels[['session_id', 'item_id', 'date']]
y_data = labels['label']
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data ,test_size = 0.3, shuffle=False)


# # Create mini labels
# trainPurchases = pd.read_csv('dressipi_recsys2022/mini_train_purchases.csv')
# trainSessions = pd.read_csv('dressipi_recsys2022/mini_train_sessions.csv')

# labels = trainPurchases.copy()
# labels['label'] = 1

# trainSessionsLabels = trainSessions.copy()
# trainSessionsLabels['label'] = 0
# labels = labels.append(trainSessionsLabels, ignore_index = True)
# labels.sort_values(by=['session_id', 'date'], inplace = True)

# labels.to_csv('dressipi_recsys2022/mini_labels.csv', index = False)


# # Split mini into train / test
# x_data = labels[['session_id', 'item_id', 'date']]
# y_data = labels['label']
# x_train, x_test, y_train, y_test = train_test_split(x_data, y_data ,test_size = 0.3, shuffle=False)
