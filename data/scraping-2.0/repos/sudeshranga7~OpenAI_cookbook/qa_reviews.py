# imports
import os

import openai
import pandas as pd
import numpy as np
from ast import literal_eval

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from openai.embeddings_utils import plot_multiclass_precision_recall, get_embedding
# load data
datafile_path = "fine_food_reviews_with_embeddings_1k.csv"

df = pd.read_csv(datafile_path)
df["embedding"] = df.embedding.apply(literal_eval).apply(np.array)  # convert string to array

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    list(df.embedding.values), df.Score, test_size=0.2, random_state=42
)

# print(list(df.embedding.values))
# train random forest classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
probas = clf.predict_proba(X_test)

report = classification_report(y_test, preds)
print(report)

# setting api key
os.environ['OPENAI_API_KEY'] = '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'
openai.api_key = os.getenv("OPENAI_API_KEY")

plot_multiclass_precision_recall(probas, y_test, [1, 2, 3, 4, 5], clf)


# predict review score real time
# Review - "Title: iphone 14 is the best phone I have ever used; Content: I switched form android.I was using samsung flagships past 10 years. But after using first ios device i can say one thing apple is the boss. It has lot of highlights such as Quality, security, camera, brand value, smoothness. overall its too good"
# Review - Title: iphone 14 pro max phone is not good; Content: I switched from android phone to an iphone. But after using first ios device i can say that iphone 14 pro max selfie camera has disappointed me a lot. Rest of the features are okay but selfie camera is a major drawback
# search_item = input("The iphone 14 is really an amazing phone, it's working so smooth from last 2 months!")
rate_review = input("Enter a review you want to rate ==> ")
product_embedding = get_embedding(
    rate_review,
    engine="text-embedding-ada-002")

pe1 = [product_embedding]

pe_score = clf.predict(pe1)
print("This review is rated "+str(pe_score[0])+" star")
