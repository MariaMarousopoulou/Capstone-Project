# Import the required libraries
import pandas as pd
from scipy.spatial.distance import pdist, squareform

# Get 'item features' dataset into a df, remove feature category column
itemFeatures = pd.read_csv('dressipi_recsys2022/item_features.csv')
newItem = itemFeatures.drop(['feature_category_id'], axis=1)

# Create cross-tabulated DataFrame from item_id and feature columns
item_cross_table = pd.crosstab(newItem['item_id'], newItem['feature_value_id'])

# Calculate all pairwise distances
jaccard_distances = pdist(item_cross_table.values, metric='jaccard')

# Convert the distances to a square matrix
jaccard_similarity_array = 1 - squareform(jaccard_distances)

# Wrap the array in a pandas DataFrame
jaccard_similarity_df = pd.DataFrame(jaccard_similarity_array,
                                     index=item_cross_table.index, columns=item_cross_table.index)

# Print the top 5 rows of the DataFrame
print(jaccard_similarity_df.head())

# Extract to csv
jaccard_similarity_df.to_csv('CBF_results', index=False)  # for full results

# Slice first 1000 rows and 1000 columns
mini_df = jaccard_similarity_df.head(1000).iloc[:, :1000]
mini_df.to_csv('CBF_results_mini_comma.csv', sep=";", decimal=",", index=False)  # for example result dataset

# Example for item 16218
# Find the values for item 16218
jaccard_similarity_series = jaccard_similarity_df.loc[16218]
# Sort these values from highest to lowest
ordered_similarities = jaccard_similarity_series.sort_values(ascending=False)
# Print the top N results
N = 10
print(ordered_similarities[:N])

# Evaluate the sensitivity / True Positive Rate of our predictions
# Get a sample of items / sessions with the purchase labeling

# Load the training datasets
trainPurchases = pd.read_csv('dressipi_recsys2022/train_purchases.csv')
trainSessions = pd.read_csv('dressipi_recsys2022/train_sessions.csv')

# Create the class label for purchases
labels = trainPurchases.copy()
labels['label'] = 1

# Create the class label for views / not purchased
trainSessionsLabels = trainSessions.copy()
trainSessionsLabels['label'] = 0

# Create a single dataset
labels = labels.append(trainSessionsLabels, ignore_index=True)
labels.sort_values(by=['session_id', 'date'], inplace=True)

# Create dataframe for evaluation
# eval_df = labels.head(9999).reset_index(drop=True).drop(['date'], axis=1)
eval_df = labels.reset_index(drop=True).drop(['date'], axis=1)
item_list = eval_df['item_id'].tolist()

# Make prediction based on item and add to list
pred_list = []
index_list = []

# Loop over item list
for i in item_list:
    series = jaccard_similarity_df.loc[i]
    y = series.sort_values(ascending=False)
    z = y[1:6].index
    x = list(y[1:6])  # exclude 1st item as it predicts the input (1:1)
    pred_list.append(x)
    index_list.append(z)

# Add lists to DataFrames and merge with eval_df
pred_df = pd.DataFrame(pred_list, columns=['Simil_1', 'Simil_2', 'Simil_3', 'Simil_4', 'Simil_5'])
index_df = pd.DataFrame(index_list, columns=['Rec_1', 'Rec_2', 'Rec_3', 'Rec_4', 'Rec_5'])
df_out = pd.concat([eval_df, index_df, pred_df], axis=1)

# Get purchased item for each session and append to df
purch_df = trainPurchases.copy()
purch_df = purch_df.drop('date', axis=1)
df_out = df_out.merge(purch_df, on='session_id', how='inner')
df_out.rename(columns={'item_id_x': 'item_id'}, inplace=True)
df_out.rename(columns={'item_id_y': 'session_purch'}, inplace=True)

df_out['check'] = 0

for index, row in df_out.iterrows():
    if row['Rec_1'] == row['session_purch'] or row['Rec_2'] == row['session_purch'] or \
       row['Rec_3'] == row['session_purch'] or row['Rec_4'] == row['session_purch'] or \
       row['Rec_5'] == row['session_purch']:
        df_out.at[index, 'check'] = 1
        
df_group = df_out.groupby(by=['session_id'])['check'].sum().to_frame()
df_group = df_group.reset_index(level=0)
truePredictions = len(df_group[(df_group['check']>0)])
totalPredictions = len(df_group)
accuratePercentage = truePredictions / totalPredictions * 100.
print('Percentage of accurate predictions: ', accuratePercentage, '%')


    