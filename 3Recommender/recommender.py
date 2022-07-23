# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 19:57:04 2022

@author: Maria_Marousopoulou
"""

import implicit
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from sklearn.preprocessing import MinMaxScaler

# read the data
df = pd.read_csv('mini_dataset.csv')
df = df.sort_values(by=['session_id', 'item_id', 'date'], ignore_index=True)

grouped_df = df.groupby(['session_id', 'item_id']).sum().reset_index()

grouped_df['session_id'] = grouped_df['session_id'].astype("category")
grouped_df['item_id'] = grouped_df['item_id'].astype("category")
grouped_df['sessionId'] = grouped_df['session_id'].cat.codes
grouped_df['itemId'] = grouped_df['item_id'].cat.codes

sparse_item_session = sparse.csr_matrix((grouped_df['label'].astype(float), (grouped_df['itemId'], grouped_df['sessionId'])))
sparse_session_item = sparse.csr_matrix((grouped_df['label'].astype(float), (grouped_df['sessionId'], grouped_df['itemId'])))

model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=50)

alpha = 15
data = (sparse_item_session * alpha).astype('double')
model.fit(data)

content_id = 1
n_similar = 10

person_vecs = model.user_factors
content_vecs = model.item_factors

content_norms = np.sqrt((content_vecs * content_vecs).sum(axis=1))

scores = content_vecs.dot(content_vecs[content_id]) / content_norms
top_idx = np.argpartition(scores, -n_similar)[-n_similar:]
similar = sorted(zip(top_idx, scores[top_idx] / content_norms[content_id]), key=lambda x: -x[1])


for content in similar:
    idx, score = content
    # print('idx:', idx)
    # print('score:', score)
    print(grouped_df.itemId.loc[grouped_df.sessionId == idx].iloc[0])
'''

def recommend(person_id, sparse_person_content, person_vecs, content_vecs, num_contents=10):
    # Get the interactions scores from the sparse person content matrix
    person_interactions = sparse_person_content[person_id, :].toarray()
    # Add 1 to everything, so that articles with no interaction yet become equal to 1
    person_interactions = person_interactions.reshape(-1) + 1
    # Make articles already interacted zero
    person_interactions[person_interactions > 1] = 0
    # Get dot product of person vector and all content vectors
    rec_vector = person_vecs[person_id, :].dot(content_vecs.T).toarray()

    # Scale this recommendation vector between 0 and 1
    min_max = MinMaxScaler()
    rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1, 1))[:, 0]
    # Content already interacted have their recommendation multiplied by zero
    recommend_vector = person_interactions * rec_vector_scaled
    # Sort the indices of the content into order of best recommendations
    content_idx = np.argsort(recommend_vector)[::-1][:num_contents]

    # Start empty list to store titles and scores
    titles = []
    scores = []

    for idx in content_idx:
        # Append titles and scores to the list
        titles.append(grouped_df.title.loc[grouped_df.content_id == idx].iloc[0])
        scores.append(recommend_vector[idx])

    recommendations = pd.DataFrame({'title': titles, 'score': scores})

    return recommendations
'''