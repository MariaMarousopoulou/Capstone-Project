# -*- coding: utf-8 -*-
"""
Created on Mon May  9 23:05:32 2022

@author: Maria_Marousopoulou
"""

import pandas as pd

candidateItems = pd.read_csv('dressipi_recsys2022/candidate_items.csv')
itemFeatures = pd.read_csv('dressipi_recsys2022/item_features.csv')
testFinalSessions = pd.read_csv('dressipi_recsys2022/test_final_sessions.csv')
testLeaderboardSessions = pd.read_csv('dressipi_recsys2022/test_leaderboard_sessions.csv')
trainPurchases = pd.read_csv('dressipi_recsys2022/train_purchases.csv')
trainSessions = pd.read_csv('dressipi_recsys2022/train_sessions.csv')

# Display Options
pd.set_option('display.width', 300)
pd.set_option('display.max_columns', None)

# Candidate Items
print('===== Canditate Items =====')
print('\nTop 5 rows')
print(candidateItems.head(5))
print('\nLast 5 rows')
print(candidateItems.tail(5))
print('\nDataframe info')
print(candidateItems.info())
print('\nDataframe shape')
print(candidateItems.shape)
print('\nDataframe size')
print(candidateItems.size)
print('\nNull values')
print(candidateItems.isnull().sum())
print('\nUnique values')
print(candidateItems.nunique())
print('\nDataframe columns')
print(candidateItems.columns)
print('\nMemory usage')
print(candidateItems.info(memory_usage="deep"))
print('\nDublicate values')
candidateItemsDublicates = candidateItems.duplicated()
print(candidateItemsDublicates.sum())


# Item Features
print('===== Item Features =====')
print('\nTop 5 rows')
print(itemFeatures.head(5))
print('\nLast 5 rows')
print(itemFeatures.tail(5))
print('\nDataframe info')
print(itemFeatures.info())
print('\nDataframe shape')
print(itemFeatures.shape)
print('\nDataframe size')
print(itemFeatures.size)
print('\nNull values')
print(itemFeatures.isnull().sum())
print('\nUnique values')
print(itemFeatures.nunique())
print('\nDataframe columns')
print(itemFeatures.columns)
print('\nMemory usage')
print(itemFeatures.info(memory_usage="deep"))
print('\nDublicate values')
itemFeaturesDublicates = itemFeatures.duplicated()
print(itemFeaturesDublicates.sum())


# Test Final Sessions 
print('===== Test Final Sessions =====')
print('\nTop 5 rows')
print(testFinalSessions.head(5))
print('\nLast 5 rows')
print(testFinalSessions.tail(5))
print('\nDataframe info')
print(testFinalSessions.info())
print('\nDataframe shape')
print(testFinalSessions.shape)
print('\nDataframe size')
print(testFinalSessions.size)
print('\nNull values')
print(testFinalSessions.isnull().sum())
print('\nUnique values')
print(testFinalSessions.nunique())
print('\nDataframe columns')
print(testFinalSessions.columns)
print('\nMemory usage')
print(testFinalSessions.info(memory_usage="deep"))
print('\nDublicate values')
testFinalSessionsDublicates = testFinalSessions.duplicated()
print(testFinalSessionsDublicates.sum())


# Test Leaderboard Sessions
print('===== Test Leaderboard Sessions =====')
print('\nTop 5 rows')
print(testLeaderboardSessions.head(5))
print('\nLast 5 rows')
print(testLeaderboardSessions.tail(5))
print('\nDataframe info')
print(testLeaderboardSessions.info())
print('\nDataframe shape')
print(testLeaderboardSessions.shape)
print('\nDataframe size')
print(testLeaderboardSessions.size)
print('\nNull values')
print(testLeaderboardSessions.isnull().sum())
print('\nUnique values')
print(testLeaderboardSessions.nunique())
print('\nDataframe columns')
print(testLeaderboardSessions.columns)
print('\nMemory usage')
print(testLeaderboardSessions.info(memory_usage="deep"))
print('\nDublicate values')
testLeaderboardSessionsDublicates = testLeaderboardSessions.duplicated()
print(testLeaderboardSessionsDublicates.sum())


# Train Purchases
print('===== Train Purchases =====')
print('\nTop 5 rows')
print(trainPurchases.head(5))
print('\nLast 5 rows')
print(trainPurchases.tail(5))
print('\nDataframe info')
print(trainPurchases.info())
print('\nDataframe shape')
print(trainPurchases.shape)
print('\nDataframe size')
print(trainPurchases.size)
print('\nNull values')
print(trainPurchases.isnull().sum())
print('\nUnique values')
print(trainPurchases.nunique())
print('\nDataframe columns')
print(trainPurchases.columns)
print('\nMemory usage')
print(trainPurchases.info(memory_usage="deep"))
print('\nDublicate values')
trainPurchasesDublicates = trainPurchases.duplicated()
print(trainPurchasesDublicates.sum())


# Train Sessions
print('===== Train Purchases =====')
print('\nTop 5 rows')
print(trainSessions.head(5))
print('\nLast 5 rows')
print(trainSessions.tail(5))
print('\nDataframe info')
print(trainSessions.info())
print('\nDataframe shape')
print(trainSessions.shape)
print('\nDataframe size')
print(trainSessions.size)
print('\nNull values')
print(trainSessions.isnull().sum())
print('\nUnique values')
print(trainSessions.nunique())
print('\nDataframe columns')
print(trainSessions.columns)
print('\nMemory usage')
print(trainSessions.info(memory_usage="deep"))
print('\nDublicate values')
trainSessionsDublicates = trainSessions.duplicated()
print(trainSessionsDublicates.sum())