# -*- coding: utf-8 -*-
"""
Created on Sun May 29 16:37:41 2022

@author: Maria_Marousopoulou
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf


# Get unique session ids
# miniSessions = pd.read_csv('mini_labels.csv')
miniSessions = pd.read_csv('../dressipi_recsys2022/labels.csv')
miniSessions = miniSessions.drop(['item_id', 'date', 'label'], axis=1)
miniSessions = miniSessions.drop_duplicates(ignore_index=True)
miniSessions = miniSessions.values.flatten().astype(int)

# Get products
products = pd.read_csv('../dressipi_recsys2022/item_features.csv')
products.drop(products.columns.difference(['item_id']), 1, inplace=True)
products = products.drop_duplicates(ignore_index=True).values.flatten().astype(int)

# Get train & test data
# miniLabels = pd.read_csv('mini_labels.csv')
miniLabels = pd.read_csv('../dressipi_recsys2022/labels.csv')
# miniLabels = miniLabels[miniLabels['label'] == 1]
miniLabels.drop(miniLabels.columns.difference(['session_id', 'item_id']), 1, inplace=True)
# train = miniLabels.head(987)
train = miniLabels.head(4020674)
train = train.reset_index()
# valid = miniLabels.tail(110)
valid = miniLabels.tail(1723146)
valid = valid.reset_index()

# Create model
sessionsEmbedding = tf.keras.layers.Embedding(len(miniSessions), 6)
productsEmbedding = tf.keras.layers.Embedding(len(products), 6)

# 40
# tf.tensordot(sessionsEmbedding(1), productsEmbedding(99), axes = [[0],[0]])

product_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(tf.constant(products, dtype=tf.int32),
                                        range(len(products))), -1)


# Create simple recommender
class SimpleRecommender(tf.keras.Model):
    def __init__(self, miniSessions, products, length_of_embedding):
        super(SimpleRecommender, self).__init__()
        self.products = tf.constant(products, dtype=tf.int32)
        # self.miniSessions = tf.constant(miniSessions, dtype=tf.string)
        self.miniSessions = tf.constant(miniSessions, dtype=tf.int32)
        self.miniSessions_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(self.miniSessions, range(len(miniSessions))), -1)
        self.product_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(self.products, range(len(products))), -1)

        self.user_embedding = tf.keras.layers.Embedding(len(miniSessions), length_of_embedding)
        self.product_embedding = tf.keras.layers.Embedding(len(products), length_of_embedding)

        self.dot = tf.keras.layers.Dot(axes=-1)

    def call(self, inputs):
        session = inputs[0]
        products = inputs[1]

        session_embedding_index = self.miniSessions_table.lookup(session)
        product_embedding_index = self.product_table.lookup(products)
        session_embedding_values = self.user_embedding(session_embedding_index)
        product_embedding_values = self.product_embedding(product_embedding_index)

        return tf.squeeze(self.dot([session_embedding_values, product_embedding_values]), 1)

    @tf.function
    def call_item_item(self, product):
        product_x = self.product_table.lookup(product)
        pe = tf.expand_dims(self.product_embedding(product_x), 0)

        all_pe = tf.expand_dims(self.product_embedding.embeddings,
                                0)  # note this only works if the layer has been built!
        scores = tf.reshape(self.dot([pe, all_pe]), [-1])

        top_scores, top_indices = tf.math.top_k(scores, k=100)
        top_ids = tf.gather(self.products, top_indices)
        return top_ids, top_scores


simpleRecommender = SimpleRecommender(miniSessions, products, 10)

sessionTensor = tf.constant([train[('session_id')].values], dtype=tf.int32)
productTensor = tf.constant([train[('item_id')].values], dtype=tf.int32)

dataset = tf.data.Dataset.from_tensor_slices((sessionTensor, productTensor))

# randomNegatives = User did not purchase
randomNegativesIndexes = tf.random.uniform((7,), minval=0, maxval=len(products), dtype=tf.int32)
tf.gather(products, randomNegativesIndexes)


class Mapper():

    def __init__(self, possible_products, num_negative_products):
        self.num_possible_products = len(possible_products)
        self.possible_products_tensor = tf.constant(possible_products, dtype=tf.int32)

        self.num_negative_products = num_negative_products
        self.y = tf.one_hot(0, depth=num_negative_products + 1)

    def __call__(self, user, product):
        randomNegativesIndexes = tf.random.uniform((self.num_negative_products,), minval=0,
                                                   maxval=self.num_possible_products, dtype=tf.int32)
        negatives = tf.gather(self.possible_products_tensor, randomNegativesIndexes)
        canditates = tf.concat([product, negatives], axis=0)
        return (user, canditates), self.y


# dataset = tf.data.Dataset.from_tensor_slices((sessionTensor, productTensor)).map(Mapper(products, 1))

# sessionNew = tf.reshape(sessionTensor, [987, 1])
sessionNew = tf.reshape(sessionTensor, [4020674, 1])
# productNew = tf.reshape(productTensor, [987, 1])
productNew = tf.reshape(productTensor, [4020674, 1])
dataset = tf.data.Dataset.from_tensor_slices((sessionNew, productNew)).map(Mapper(products, 10))

for (u, c), y in dataset:
    print(u)
    print(c)
    print(y)
    break


def get_dataset(df, products, num_negative_products):
    sessionTensor = tf.constant([df[('session_id')].values], dtype=tf.int32)
    productTensor = tf.constant([df[('item_id')].values], dtype=tf.int32)
    sessionNew = tf.reshape(sessionTensor, [sessionTensor.shape[1], 1])
    productNew = tf.reshape(productTensor, [productTensor.shape[1], 1])
    dataset = tf.data.Dataset.from_tensor_slices((sessionNew, productNew))
    dataset = dataset.map(Mapper(products, num_negative_products))
    dataset = dataset.batch(2)
    return dataset


for (u, c), y in get_dataset(train, products, 10):
    print(u)
    print(c)
    print(y)
    break

model = SimpleRecommender(miniSessions, products, 15)
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.SGD(learning_rate=100.),
              metrics=[tf.keras.metrics.CategoricalAccuracy()])

model.fit(get_dataset(train, products, 100), validation_data=get_dataset(valid, products, 100), epochs=5)

# print("Recs for item {}: {}".format(4026, model.call_item_item(tf.constant(4026, dtype=tf.int32))))
print("Recs for item {}: {}".format(26, model.call_item_item(tf.constant(26, dtype=tf.int32))))

# dataset = tf.data.Dataset.from_tensor_slices((sessionTensor, productTensor)).map(Mapper(products, 10))
dataset = tf.data.Dataset.from_tensor_slices((sessionNew, productNew)).map(Mapper(products, 10))

dataset = dataset.batch(32)
model.evaluate(dataset)
