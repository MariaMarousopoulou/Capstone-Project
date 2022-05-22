import pandas as pd
import numpy as np

trainPurchases = pd.read_csv('dressipi_recsys2022/train_purchases.csv')
trainSessions = pd.read_csv('dressipi_recsys2022/train_sessions.csv')

labels = trainPurchases.copy()
labels['label'] = 1

trainSessionsLabels = trainSessions.copy()
trainSessionsLabels['label'] = 0

labels = labels.append(trainSessionsLabels, ignore_index = True)
labels.sort_values(by=['session_id', 'date'], inplace = True)

'''
for i in range(1, 74):
    labels[i] = np.nan
    
for index, row in labels.iterrows():
    d2 = itemFeatures[itemFeatures['item_id'] == row['item_id']]

pd.merge(pd.melt(labels, id_vars='item_id'), itemFeatures,left_on=['feature_category_id', 'feature_value_id'], \
         right_on=['feature_category_id', 'feature_value_id']).pivot('item_id', 'feature_category_id', 'feature_value_id')

df = (labels.set_index('item_id').rename_axis('feature_category_id', axis=1).stack()\
      .reset_index(name='feature_value_id').merge(itemFeatures)\
      .set_index(['item_id', 'feature_category_id']).weight.unstack())
      
labels.item_id.astype(object)

df = pd.merge(pd.melt(labels, id_vars='item_id'), itemFeatures,left_on=['item_id', 'feature_value_id'], \
         right_on=['feature_category_id', 'feature_value_id']).pivot('item_id', 'feature_category_id', 'feature_value_id')      
      
'''

itemFeatures = pd.read_csv('dressipi_recsys2022/item_features.csv')
df = pd.pivot_table(itemFeatures, index='item_id', columns='feature_category_id')
newdf = labels.merge(df, left_on='item_id', right_on='item_id')
