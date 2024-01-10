# %%
from caafe import CAAFEClassifier # Automated Feature Engineering for tabular datasets
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
import os
import openai
from caafe import data
from tabpfn.scripts import tabular_metrics
from sklearn.metrics import accuracy_score
import sys
sys.path.append('../')
from SMARTFEAT.Prediction_helper import *
import time
import pandas as pd 
from SMARTFEAT.serialize import *
import numpy as np
from sklearn.model_selection import train_test_split
# %%
openai.api_key = "YOUR_OPENAI_APIKEY"
metric_used = tabular_metrics.auc_metric
# %% load input dataset
data_df = pd.read_csv("../dataset/[DatasetPath]/[DatasetWithNewFeatures].csv")
y_label = 'Y_Label'
# %% data preprocessing
data_df, features = data_preproessing(data_df, y_label)
X = data_df[features]
y = data_df[y_label]

# %% split dataset
X_train, X_test, y_train, y_test =train_test_split(data_df[features],data_df[y_label],
                                                   test_size=0.25,
                                                   random_state=0,
                                                   stratify=data_df[y_label])
df_train = pd.concat([X_train,y_train], axis=1)
df_test = pd.concat([X_test,y_test], axis=1)

#%% before feature engineering
models = GetBasedModel()
names,results, tests = PredictionML(X_train, y_train,X_test, y_test,models)
basedLineScore = ScoreDataFrame(names,results, tests)
basedLineScore

#%% Setup and Run CAAFE for the five classifiers
clf_no_feat_eng = LogisticRegression()
# clf_no_feat_eng = GaussianNB()
# clf_no_feat_eng = RandomForestClassifier()
# clf_no_feat_eng = ExtraTreesClassifier()
# clf_no_feat_eng = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', learning_rate='constant', learning_rate_init=0.0001, alpha=0.001, max_iter=1000)

start_time = time.time()
with open('[PathtoFolder]/dataset/[DatasetPath]/data_agenda.txt', 'r') as f:
    data_agenda = f.read()
f.close
caafe_clf = CAAFEClassifier(base_classifier=clf_no_feat_eng,
                            llm_model="gpt-4",
                            iterations=10)
caafe_clf.fit_pandas(df_train,
                     target_column_name=y_label,
                     dataset_description=data_agenda)
pred = caafe_clf.predict(df_test)
end_time = time.time()
print("The total timeoverhead is")
print(end_time-start_time)

# %% print the code
print(caafe_clf.code)

# %%
X_train, X_test, y_train, y_test =train_test_split(data_df[features],data_df[y_label],
                                                   test_size=0.25,
                                                   random_state=0,
                                                   stratify=data_df[y_label])

# apply the code to both the X_train and X_test
'''
Python code given in caafe_clf.code
'''
# %% AUC score after data remedy
from sklearn.metrics import roc_auc_score
clf_no_feat_eng.fit(X_train, y_train)
test_score = roc_auc_score(y_test, clf_no_feat_eng.predict_proba(X_test)[:, 1])
print(test_score)

# %% evaluate feature usefulness.
import sys
sys.path.append('../')
import pathlib as Path
from SMARTFEAT.serialize import *
from SMARTFEAT.feature_evaluation import feature_evaluation_show_all, feature_evaluation_select_k
import pandas as pd 
print("===========================================")
print('mutual info')
feature_evaluation_show_all(X_train, y_train, 'mutual info')
print("===========================================")
print('rfe-rf')
feature_evaluation_show_all(X_train, y_train, 'rfe-rf')
print("===========================================")    
print('feature_importance')
feature_evaluation_show_all(X_train, y_train, 'feature_importance')
print("===========================================")    
# %%
