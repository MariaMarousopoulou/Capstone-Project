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
jaccard_similarity_df = pd.DataFrame(jaccard_similarity_array, index=item_cross_table.index,
                                     columns=item_cross_table.index)

# Print the top 5 rows of the DataFrame
print(jaccard_similarity_df.head())

# Extract to csv
jaccard_similarity_df.to_csv('CBF_results.csv', index=False)  # for full results

# Slice first 1000 rows and 1000 columns
mini_df = jaccard_similarity_df.head(1000).iloc[:, :1000]
mini_df.to_csv('CBF_results_mini.csv', index=False)  # for example result dataset

# Example for item 16218
# Find the values for item 16218
jaccard_similarity_series = jaccard_similarity_df.loc[16218]
# Sort these values from highest to lowest
ordered_similarities = jaccard_similarity_series.sort_values(ascending=False)
# Print the results (first N = 10)
print(ordered_similarities[:10])
