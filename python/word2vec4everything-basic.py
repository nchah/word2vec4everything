#!/usr/bin/env python
# -*- coding: utf-8 -*-

# word2vec4everything.py 
# A modified version of the starter code provided in the TensorFlow tutorial.
#
# Run: $ python word2vec4everything.py [/path/to/file]
# ==============================================================================

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import codecs
import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import nltk
import string


# Setting up the program's arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input_data', help='Path to the input data file')
# parser.add_argument('--', help='')
parser.add_argument('--train_steps', type=int, 
                    default=20000, help='Number of training steps.')
parser.add_argument('--vocab_size', type=int, 
                    default=5000, help='Number of words in the dictionary.')
parser.add_argument('--plot_count', type=int, 
                    default=1500, help='Number of points to consider in visual.')
parser.add_argument('--whitelist_labels', type=str,  # Convert to list downstream 
                    help='')
# parser.add_argument('--', help='')
args = parser.parse_args()


# Step 1: Load the data.
def read_data(input_data):
    """Load the dataset"""
    with codecs.open(input_data, encoding='utf8') as f:
        data = f.read()
    return data

data = read_data(args.input_data)

# Pre-procssing: Stopwords and Punctuation
stopwords = nltk.corpus.stopwords.words('english')
punctuation = set(string.punctuation)
punctuation.remove("'")
for p in punctuation:
    data = data.replace(p, '')
    print('replaced: ' + p)

words = tf.compat.as_str(data).split()
print('Data size:', len(words))


# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = args.vocab_size

def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


data, count, dictionary, reverse_dictionary = build_dataset(words)
del words  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]],
          '->', labels[i, 0], reverse_dictionary[labels[i, 0]])


# Step 4: Build and train a skip-gram model.
batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64  # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():
    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_mean(
            tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                           num_sampled, vocabulary_size))

    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True)

    # Add variable initializer.
    init = tf.initialize_all_variables()


# Step 5: Begin training.
num_steps = args.train_steps

with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    init.run()
    print("Initialized")

    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(
                batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = "Nearest to %s:" % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    lose_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()


# Step 6: Visualize the embeddings.
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(12, 12))  # in inches
    texts = []
    # Explicitly set the labels to graph from the source text
    whitelist_labels = args.whitelist_labels.split(',')
    for i, label in enumerate(labels):
        pos = nltk.pos_tag([label])
        # List POS tags with >>> nltk.help.upenn_tagset()
        ignore_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'LS', 'MD', 'PDT', 'PRP', 'PRP$',
        'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 
        'WDT', 'WP', 'WP$', 'WRB']
        # Logic for selecting the labels to visualize
        if label.lower() not in stopwords \
            and label[0].isupper() \
            and pos[0][1] not in ignore_tags \
            and "'" not in label and u"’" not in label:
            x, y = low_dim_embs[i, :]
            texts.append(plt.text(x, y, label))
            plt.scatter(x, y)
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))  # For label placements
            # plt.annotate(label,
                         # xy=(x, y),
                         # xytext=(5, 2),
                         # textcoords='offset points',
                         # ha='right',
                         # va='bottom')
    plt.savefig(filename, dpi=600)
    subprocess.call(["say 'program completed'"], shell=True)  # Audio notification for OS X =)


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import nltk
from nltk import pos_tag
from adjustText import adjust_text
import subprocess

stopwords = nltk.corpus.stopwords.words('english')

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = args.plot_count
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])

labels = [reverse_dictionary[i] for i in xrange(plot_only)]
labels = [unicode(word, 'utf-8') for word in labels]
print(str(labels))
# print(len(low_dim_embs), len(labels))

plot_with_labels(low_dim_embs, labels)
