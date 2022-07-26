# Import the required libraries
from scipy.sparse.linalg import svds
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Set display precision / No. of decimals
pd.options.display.precision = 10

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

# Create mini dataframe for preliminary work or take the full dataset
mini_df = labels.head(100000).iloc[:, :2000].reset_index(drop=True)
full_df = labels.reset_index(drop=True)

# Add features to get rating (final_score)
# Each single feature contributes to a total

# Purchased or not (+3 to total rating)
full_df['purchased'] = 3 * full_df['label']

# Viewed or not (+1 to total rating)
full_df['viewed'] = 1

# How many times viewed (+0.5 for one view, increased with each additional view, with asymptomatic limit to +1)
full_df['combined_session_item'] = full_df['session_id'].astype(str) + full_df['item_id'].astype(str)
full_df['views_in_session'] = full_df.groupby(['combined_session_item'])['combined_session_item'].transform('count')
full_df['score_views'] = (0.5 - (0.25 / (full_df['views_in_session'])**full_df['views_in_session']))*2

# Fix timestamp format / dtype
full_df['date_time'] = pd.to_datetime(full_df['date'])

# View duration in session vs session duration (+0 to +1 based on view duration as pct of total session)
full_df['view_duration'] = full_df['date_time'].shift(-1) - full_df['date_time']
full_df['seconds'] = full_df['view_duration'].dt.total_seconds().round(0)
full_df['sec'] = abs(full_df['seconds'].round(0))
full_df.loc[full_df['sec'] > 86400, 'sec'] = 0
full_df.loc[full_df['sec'] == 0, 'sec'] = full_df.groupby('session_id', as_index=False)['sec'].median()
full_df['session_duration'] = full_df.groupby('session_id')['sec'].transform('sum')
full_df['view_dur_rating'] = round(full_df['sec'].div(full_df['session_duration'].values), 4)

# Times purchased in dataset (inverse - the fewest purchases the better - pct based)
full_df['tot_item_purch'] = full_df.groupby(['item_id'])['item_id'].transform('count')
full_df['purch_pct_rating'] = full_df['label'] / full_df['tot_item_purch']

# Times viewed in dataset vs total (inverse - the fewest views the better - pct based) up to +1
full_df['views_in_dataset'] = full_df.groupby(['item_id'])['item_id'].transform('count')
full_df['view_pct_rating'] = round(full_df['views_in_session'] / full_df['views_in_dataset'], 4)

# Total session time to percentage of max session time (up to +1)
full_df['session_dur_rating'] = full_df['session_duration'] / full_df['session_duration'].max()

# Ιtem view duration to max item view duration (reverse - shortest view time)
full_df['item_view_dur_rating'] = full_df['sec'] / full_df['sec'].max()

# Total rating (sum of all individual ratings - up to 10)
full_df['final_score'] = full_df['purchased'] + full_df['viewed'] + full_df['score_views'] + full_df['view_dur_rating']\
                         + full_df['purch_pct_rating'] + full_df['view_pct_rating'] + full_df['session_dur_rating']\
                         + full_df['item_view_dur_rating']

# Create new dataframe with only session, item, and score data / Remove duplicates
session_ratings_df = full_df[['session_id', 'item_id', 'final_score']].copy().reset_index(drop=True)
session_ratings_df = session_ratings_df.sort_values(by=['session_id', 'item_id', 'final_score'])
session_ratings_df = session_ratings_df[['session_id', 'item_id', 'final_score']].copy().reset_index(drop=True)
session_ratings_df['combined_session_item'] = session_ratings_df['session_id'].astype(str) \
                                              + session_ratings_df['item_id'].astype(str)
session_ratings_df2 = session_ratings_df.drop_duplicates('combined_session_item', keep='last')
session_ratings_df3 = session_ratings_df2[['session_id', 'item_id', 'final_score']].copy().reset_index(drop=True)

# Transform the table
session_ratings_table = session_ratings_df3.pivot(index='session_id', columns='item_id', values='final_score')

# Count the occupied cells to identify the sparsity of the dataset
sparsity_count = session_ratings_table.isnull().values.sum()

# Count all cells
full_count = session_ratings_table.size

# Find the sparsity of the DataFrame
sparsity = sparsity_count / full_count
print(sparsity)

# Get the average rating for each session
avg_ratings = session_ratings_table.mean(axis=1)

# Center each session ratings around 0
session_ratings_table_centered = session_ratings_table.sub(avg_ratings, axis=1)

# Fill in the missing data with 0s
session_ratings_table_normed = session_ratings_table_centered.fillna(0)

# Decompose the matrix
U, sigma, Vt = svds(session_ratings_table_normed)

# Convert sigma into a diagonal matrix
sigma = np.diag(sigma)
print(sigma)

# Dot product of U and sigma
U_sigma = np.dot(U, sigma)

# Dot product of result and Vt
U_sigma_Vt = np.dot(U_sigma, Vt)

# Print the result
print(U_sigma_Vt)

# Add back on the row means contained in avg_ratings
uncentered_ratings = U_sigma_Vt + avg_ratings.values.reshape(-1, 1)

# Create DataFrame of the results
calc_pred_ratings_df = pd.DataFrame(uncentered_ratings, index=session_ratings_table_normed.index,
                                    columns=session_ratings_table_normed.columns)

# Print example of the recalculated matrix
calc_pred_ratings_df.head()

# Sort the ratings of session 3 from high to low and print the top N recommendations
N = 5
session_3_ratings = calc_pred_ratings_df.loc[3, :].sort_values(ascending=False)
print(session_3_ratings[:5])

# Evaluate rating predictions / Root Mean Squared Error

# Establish a base to compare against our predicted ratings
actual_values = session_ratings_table_centered.iloc[:, :].values
predicted_values = calc_pred_ratings_df.iloc[:, :].values

# Create a mask of actual_values to only look at the non-missing values in the ground truth
mask = ~np.isnan(actual_values)

# Print the performance of both predictions and compare
print("RMSE is", round(mean_squared_error(actual_values[mask], predicted_values[mask], squared=False), 3))
