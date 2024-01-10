from nltk.corpus import reuters
import openai
from tqdm import tqdm
import numpy as np
import nltk
from openai.embeddings_utils import get_embeddings
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle

with open('/users/jasper/oai.txt', 'r') as f:
    openai.api_key = f.read()

# nltk.download('reuters')

# perform train/test split
train_docs_id = reuters.fileids(categories='trade')
test_docs_id = reuters.fileids(categories='crude')

# get train/test docs
train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id]
test_docs = [reuters.raw(doc_id) for doc_id in test_docs_id]

# get train/test labels
train_labels = [reuters.categories(doc_id)[0] for doc_id in train_docs_id]
test_labels = [reuters.categories(doc_id)[0] for doc_id in test_docs_id]

# get embeddings for train/test docs
# print('Getting embeddings...')
# train_embeddings = get_embeddings(train_docs)
# test_embeddings = get_embeddings(test_docs)

# # pickle embeddings
# with open('embeddings/train_embeddings.pkl', 'wb') as f:
#     pickle.dump(train_embeddings, f)
# with open('embeddings/test_embeddings.pkl', 'wb') as f:
#     pickle.dump(test_embeddings, f)

# load embeddings
with open('embeddings/train_embeddings.pkl', 'rb') as f:
    train_embeddings = pickle.load(f)
with open('embeddings/test_embeddings.pkl', 'rb') as f:
    test_embeddings = pickle.load(f)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(train_embeddings, train_labels, test_size=0.2, random_state=42)

# train classifier
clf = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)
clf.fit(X_train, y_train)

# predict on test set
y_pred = clf.predict(X_test)

# evaluate
print(accuracy_score(y_test, y_pred))

import code
code.interact(local=locals())