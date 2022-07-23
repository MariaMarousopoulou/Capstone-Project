import pandas as pd
import numpy as np

trainPurchases = pd.read_csv('dressipi_recsys2022/train_purchases.csv')
trainSessions = pd.read_csv('dressipi_recsys2022/train_sessions.csv')

labels = trainPurchases.copy()
labels['label'] = 1

trainSessionsLabels = trainSessions.copy()
trainSessionsLabels['label'] = 0

labels = labels.append(trainSessionsLabels, ignore_index=True)
labels.sort_values(by=['session_id', 'date'], inplace=True)

'''
for i in range(1, 74):
    labels[i] = np.nan
    
for index, row in labels.iterrows():
    d2 = itemFeatures[itemFeatures['item_id'] == row['item_id']]

pd.merge(pd.melt(labels, id_vars='item_id'), itemFeatures,left_on=['feature_category_id', 'feature_value_id'],\
         right_on=['feature_category_id', 'feature_value_id']).pivot('item_id', 'feature_category_id',\
         'feature_value_id')

df = (labels.set_index('item_id').rename_axis('feature_category_id', axis=1).stack()\
      .reset_index(name='feature_value_id').merge(itemFeatures)\
      .set_index(['item_id', 'feature_category_id']).weight.unstack())
      
labels.item_id.astype(object)

df = pd.merge(pd.melt(labels, id_vars='item_id'), itemFeatures,left_on=['item_id', 'feature_value_id'],\
         right_on=['feature_category_id', 'feature_value_id']).pivot('item_id', 'feature_category_id',\
         'feature_value_id')      
      
'''

itemFeatures = pd.read_csv('dressipi_recsys2022/item_features.csv')
df = pd.pivot_table(itemFeatures, index='item_id', columns='feature_category_id')
newdf = labels.merge(df, left_on='item_id', right_on='item_id')

# extract
newdf.to_csv('dataset.csv', index=False)  # don't
# newdf.head(200000).to_csv('minidata.csv', index=False)


# creating new custom features
df1 = newdf[['session_id', 'date']]
# print(df.head())

# converting to datetimes
df1['date'] = pd.to_datetime(df1['date'], format='%Y-%m-%d %H:%M:%S.%f')
df1.sort_values(by=['session_id', 'date'], inplace=True)

# grouping per 1Min and id
g = df1.groupby(['session_id', pd.Grouper(key="date", freq='1min', origin='start')])

# get first values per groups to new column
df1['new'] = g['date'].transform('first')
# subtract by timestamp and convert timedeltas to seconds
df1['new'] = df1['date'].sub(df1['new']).dt.total_seconds()
# shifting per groups by id
df1['old'] = df1.groupby('session_id')['new'].shift()
# get first value per groups, now shifted
df1['old'] = g['old'].transform('first')
# replace 0 to misisng values and get average
df1['avg_time'] = df1[['old', 'new']].replace(0, np.nan).mean(axis=1).fillna(df1['new'])

# create helper columns defining contiguous blocks and day
df1['daily_view'] = (df1['new'].astype(bool).shift() != df1['new'].astype(bool)).cumsum()
df1['days'] = df1['date'].dt.normalize()

# group by day to get unique block count and value count
session_map = df1[df1['new'].astype(bool)].groupby('days')['daily_view'].nunique()
seconds_map = df1[df1['new'].astype(bool)].groupby('days')['new'].count()

# map to original dataframe
df1['sessions'] = df1['days'].map(session_map)
df1['seconds'] = df1['days'].map(seconds_map)

# calculate result
res = df1.groupby(['days', 'seconds', 'sessions'], as_index=False)['avg_time'].sum()
print(res)
