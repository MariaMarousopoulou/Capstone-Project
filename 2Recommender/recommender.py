# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 19:13:19 2022

@author: Maria_Marousopoulou
"""

import os
import implicit
import pandas as pd
import seaborn as sns
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# read the data
# data = pd.read_csv('mini_dataset.csv')
data = pd.read_csv('../dressipi_recsys2022/dataset.csv')
data = data.sort_values(by=['session_id', 'item_id', 'date'], ignore_index=True)
# print("DataFrame is:\n", data.head())

# print('Number of rows and columns:', data.shape)

# separate the datetime to date and time
data['Date'] = pd.to_datetime(data['date']).dt.date
data['Time'] = pd.to_datetime(data['date']).dt.time
# print("Date-time-hour-minutes :\n", data)


# # making observations for datetime
# df = data.Date.value_counts()
# print('Number of views on different days:\n', df)
# df1 = data.Time.value_counts()
# print('Number of views on different hours of the day:\n', df1)


# data.Date = data.Date.astype('str')
# order = data.Date.value_counts().index
# color = sns.color_palette()[9]

print('Number of missing values:\n', data.isnull().sum())
data = data.fillna(0)

# data.to_csv('datasetDates.csv')

# model needs session_id & item_id as numeric. So convert into numeric.
# data['session_id'] = data['session_id'].astype('category').cat.codes
# data['item_id'] = data['item_id'].astype('category').cat.codes
data['sessionId'] = data['session_id'].astype('category').cat.codes
data['itemId'] = data['item_id'].astype('category').cat.codes

# retain old session_id & item_id so generate recommended list based on old values
train, cros_val = train_test_split(data, test_size=0.2, random_state=1)
train, test = train_test_split(train, test_size=0.25, random_state=1)

print('Splitted dataset into train set, test set and cross validation set:')
print('Train shape:', train.shape, 'Percentage:', train.shape[0] / data.shape[0] * 100, '%')
print('Test shape:', test.shape, 'Percentage:', test.shape[0] / data.shape[0] * 100, '%')
print('Cross validation shape:', cros_val.shape, 'Percentage:', cros_val.shape[0] / data.shape[0] * 100, '%')

# user_items = sparse.csr_matrix((train["('feature_value_id', 1)"].astype(float), (train['session_id'], train['item_id']))
#                                )
# item_users = sparse.csr_matrix((train["('feature_value_id', 1)"].astype(float), (train['item_id'], train['session_id']))
#                                )

user_items = sparse.csr_matrix((train["('feature_value_id', 1)"].astype(float), (train['sessionId'], train['itemId']))
                               )
item_users = sparse.csr_matrix((train["('feature_value_id', 1)"].astype(float), (train['itemId'], train['sessionId']))
                               )

os.environ['MKL_NUM_THREADS'] = '1'  # To avoid multithreading.
os.environ['OPENBLAS_NUM_THREADS'] = '1'

model = implicit.als.AlternatingLeastSquares(factors=500, iterations=10)

alpha = 40
train_conf = (item_users * alpha).astype('double')

# model.fit(train_conf)

import csv

fields = 'sessionId', 'item_list'
filename = 'rec_train.csv'
with open(filename, 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(fields)
    userId = train['sessionId'].values.tolist()
    for user in userId:
        scores = []
        items = []
        results = []
        results.append(user)
        recommendations = model.recommend(user, user_items[user], N=5)
        print('ok')
        for item in recommendations:
            ids, score = item
            scores.append(score)
            items.append(ids)
        results.append(items)
        writer.writerow(results)
