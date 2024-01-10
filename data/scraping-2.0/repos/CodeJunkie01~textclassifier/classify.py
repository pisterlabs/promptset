# imports
import os
import pandas as pd
import numpy as np
import datetime
from pyzotero import zotero
from decouple import config
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
from clearScreen import cls
from imblearn.ensemble import BalancedRandomForestClassifier
import time
import openai
from decouple import config

openai.api_key = config('OPENAI_SECRET')
openai.organization = config('OPENAI_ORG_ID')


def createZot():
    library_type = config('ZOTERO_LIBRARY_TYPE')
    library_id = config('ZOTERO_USER_ID')
    api_key = config('ZOTERO_KEY')
    return zotero.Zotero(library_id, library_type, api_key)


cls()
print("Starting classification...")
version = 0
with open('currentVersion.txt', 'r') as f:
    lines = f.readlines()
    version = int(lines[0])
# load data
trainingdata_path = "csv/training" + version.__str__() + ".csv"
prediction_path = "csv/prediction.csv"
cls()
print("Loading data...")
df = pd.read_csv(trainingdata_path)
df["embedding"] = df.embedding.apply(eval).apply(
    np.array)  # convert string to array
df["relevant"] = df.relevant.apply(
    lambda x: 1 if x else 0)  # convert bool to int
# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    list(df.embedding.values), df.relevant, test_size=0.2, random_state=42
)

# train random forest classifier
cls()
print("Training classifier...")
# clf = RandomForestClassifier(max_depth=5, max_features=0.8, max_samples=0.5,  # type: ignore
#                             n_estimators=170)
# clf = BalancedRandomForestClassifier(max_depth=5, max_features=0.9, max_samples=0.5,
#                                     n_estimators=110)
clf = BalancedRandomForestClassifier(max_depth=9, max_features=0.8, max_samples=0.3,
                                     n_estimators=185)
clf.fit(X_train, y_train)
df = pd.read_csv(prediction_path)
df["embedding"] = df.embedding.apply(eval).apply(
    np.array)  # convert string to array
cls()
print("Predicting relevance...")
relevancePrediction = clf.predict(list(df.embedding.values))
relevancePredictionProbas = clf.predict_proba(list(df.embedding.values))
# for i in range(len(relevancePrediction)):
#    if relevancePrediction[i] == 1:
#       print(i.__str__() + "- relevant: " + df["title"][i])
#   else:
#       print(i.__str__() + "- not relevant: " + df["title"][i])
date = datetime.datetime.now()
save = False
inMenu = True
i = -1
inPointMenu = True
renderAbstract = False
while inMenu:
    inPointMenu = True
    if (i < len(relevancePrediction)-1):
        i += 1
    page = i+1
    indexString = page.__str__() + "/" + len(relevancePrediction).__str__()
    if (page < 10):
        indexString = "0" + indexString
    while inPointMenu:
        cls()
        print("ENTER:continue - a:show abstract - c:change prediction - s:save - q:quit - p:previous - e:export")
        print("------------ "+indexString+" ------------")
        # if relevancePrediction[i] == 1:
        #    print('\033[92m' + df["title"][i] + '\033[0m')
        if relevancePredictionProbas[i][1] >= 0.75:
            print('\033[92m' + df["title"][i] + '\033[0m')
        elif relevancePredictionProbas[i][1] >= 0.5:
            print('\033[93m' + df["title"][i] + '\033[0m')
        else:
            print(df["title"][i])
        if renderAbstract:
            print("")
            print(df["abstract"][i])
            renderAbstract = False
        action = input()
        if action == "a":
            print(df["abstract"][i])
            renderAbstract = True
        elif action == "":
            inPointMenu = False
        elif action == "c":
            prevPred = relevancePrediction[i]
            if (prevPred == 1):
                relevancePrediction[i] = 0
                relevancePredictionProbas[i][1] = 0
            else:
                relevancePrediction[i] = 1
                relevancePredictionProbas[i][1] = 1
        elif action == "s":
            action = input("Are you sure you want to save? (y/n)")
            if action == "y":
                save = True
                inMenu = False
                inPointMenu = False
                break
        elif action == "q":
            action = input("Are you sure you want to quit? (y/n)")
            if action == "y":
                inMenu = False
                inPointMenu = False
                break
        elif action == "p":
            if i > 0:
                i -= 1
        elif action == "e":
            # add item to zotero collection via its id
            action2 = input("Are you sure you want to export? (y/n)")
            if action2 != "y":
                continue
            zot = createZot()
            exportCount = 0
            for y in range(len(relevancePrediction)):
                if relevancePrediction[y] == 1:
                    exportCount += 1

            for x in range(len(relevancePrediction)):
                if relevancePredictionProbas[x][1] >= 0.75:
                    id = df["id"][x]
                    item = zot.item(id)
                    updated = zot.add_tags(item, ["likely relevant"])
                    cls()
                    print("Adding tags: [" + "#"*int((x+1)/len(relevancePrediction)*40) + " "*(
                        40-int((x+1)/len(relevancePrediction)*40)) + "]")
                elif relevancePredictionProbas[x][1] >= 0.5:
                    id = df["id"][x]
                    item = zot.item(id)
                    updated = zot.add_tags(item, ["maybe relevant"])
                    cls()
                    print("Adding tags: [" + "#"*int((x+1)/len(relevancePrediction)*40) + " "*(
                        40-int((x+1)/len(relevancePrediction)*40)) + "]")
            cls()
            print("Exported " + exportCount.__str__() +
                  " items to zotero. Resuming...")
            time.sleep(2)
        elif action == "i":
            action2 = input("Are you sure you want to improve? (y/n)")
            if action2 != "y":
                continue
            i = -1
            zot = createZot()
            for pred in relevancePredictionProbas:
                i = i+1
                if pred[1] <= 0.75 and pred[1] >= 0.5:
                    completion = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a classifier that predicts whether a paper is relevant based on a prompt. Only ever answer with 'relevant' or 'not relevant'."},

                            {"role": "user", "content": "Is this abstract relevant in the context of machine learning, business intelligence and analytics? The paper is only relevant, if the topics were actually studied, not if they were used to study a different topic. ABSTRACT:" +
                                df["abstract"][i]},
                        ],
                    )
                    print(df["abstract"][i])
                    print(completion.choices[0].message)
                    if completion.choices[0].message.content.lower() == "relevant":
                        id = df["id"][i]
                        item = zot.item(id)
                        updated = zot.add_tags(
                            item, ["relevant according to llm2"])

if save:
    cls()
    print("Saving...")
    trainingData = pd.read_csv(trainingdata_path)
    for index in range(len(relevancePrediction)):
        new_row = pd.Series({'title': df["title"][index], 'abstract': df["abstract"][index],
                            'authors': df["authors"][index], 'embedding': list(df["embedding"][index]), 'date': date, 'relevant': relevancePrediction[index]})
        trainingData = pd.concat(
            [trainingData, new_row.to_frame().T], ignore_index=True)
    trainingData.to_csv('csv/training' + (version+1).__str__() + '.csv',
                        index=False, header=True)
    with open('currentVersion.txt', 'w') as f:
        f.write((version+1).__str__())


preds = clf.predict(X_test)
probas = clf.predict_proba(X_test)
report = classification_report(y_test, preds)


print(report)
