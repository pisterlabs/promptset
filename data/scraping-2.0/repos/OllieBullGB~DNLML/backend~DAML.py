import os
import pandas as pd
import numpy as np
import openai 
import pickle
import uuid
import ast
from dotenv import load_dotenv

from sklearn.preprocessing import PowerTransformer, LabelEncoder
from sklearn.model_selection import train_test_split as tt_split
from sklearn.metrics import accuracy_score, mean_squared_error as mse
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import SGDClassifier, Ridge
from sklearn.kernel_approximation import RBFSampler

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")



class DAML:
  def __init__(self, df, task, verbose = 0, forceTask=None):
    self.df = df
    if forceTask is None:
      self.task, self.target_column = self.process_NLP_task(task)
      if verbose == 1:
        print("TASK:", self.task)
        print("TARGET COLUMN:", self.target_column)
    else:
      self.task, self.target_column = forceTask
    self.models = []
    self.verbose = verbose

    # find target column based off the task.targetColumn after NLP processing

    self.clean()

    # Split data into 60% training, 20% validation, 20% test.
    # Ensure that data is the same across all models for accurate comparison
    X = self.df.drop(self.target_column, axis=1)
    y = self.df[self.target_column]

    X_temp, self.X_test, y_temp, self.y_test = tt_split(X, y,test_size=0.2, random_state=42)
    self.X_train, self.X_val, self.y_train, self.y_val = tt_split(X_temp, y_temp, test_size = 0.25, random_state=42)

    self.selected_model = None
    self.model()

    # sort the models by accuracy desc
    reverse_order = self.task != 'regression'
    self.models.sort(key=lambda x: x[1], reverse=reverse_order)

    # pick the type of the best model to be selected_model
    self.selected_model = self.models[0][0].__class__.__name__
    self.selected_model_url = None

    # dump best model
    if len(self.models) > 0:
      domain_name = "https://python-nlai-service.onrender.com/"
      model_name = str(uuid.uuid4()) + "model.sav"
      pickle.dump(self.models[0][0], open(f'./models/{model_name}', 'wb'))
      self.selected_model_url = domain_name + "models/" + model_name

  def process_NLP_task(self, task, processor="open-ai"):
    column_string = ', '.join(list(self.df.columns))
    query = f"{task}. The columns for my dataset are {column_string}"
    message = "You are a natural language processor that can talk exclusively in JSON. You are tasked with analysing a query for a dataset. For example 'I want to classify more results based on an income of >=50k usd. The columns for my dataset are 'age','profession','nationality','income'' would return {'task': 'classification (labelled)', 'targetColumn':'income}'. Target column must exist within the given set of columns, task must be one of 'classification (unlabelled) or 'regression'. Your task is: " + query

    if processor == "open-ai":
      messages = [{"role":"user", "content": message}]
      chat = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages = messages)
      reply = chat.choices[0].message.content
      response = ast.literal_eval(reply)
      return (response['task'], response['targetColumn'])
    else:
      return ("classification (labelled)", "price_range")

  def clean(self):
    # eliminate NaN values
    # ->> drop records/columns
    # ->> impute values
    # normalise values
    # encode categorical data

    # deal with missing values
    self.df.dropna(axis=0, inplace=True)
    # encode categorical data
    cols = self.df.columns
    numerical_cols = self.df._get_numeric_data().columns
    categorical_cols = list(set(cols) - set(numerical_cols))
    for col_name in categorical_cols:
      le = LabelEncoder()
      self.df[col_name] = le.fit_transform(self.df[col_name])


    # normalise value
    transformer = PowerTransformer(method='yeo-johnson')
    X_cols = list(self.df.columns)
    X_cols.remove(self.target_column)

    for col in X_cols:
      self.df[col] = transformer.fit_transform(self.df[col].to_numpy().reshape(-1, 1))


  def model(self, big_dataset_size=100000):
    if self.task == 'classification (labelled)':
      print('classification (labelled)')
      # <100k instances
      # yes:
      # --> LinearSVC, KNeighbors, SVC, Ensemble Classifiers
      # no:
      # --> SGD Classifier, Kernel Approximation
      num_records = self.df.shape[0]
      if num_records < big_dataset_size:
        print("< 100k")
        self.init_linearSVC()
        self.init_knn()
        self.init_SVC()
        self.init_RandomForestClassifier()
        # Start a LinearSVC, KNeighbors, SVC, Ensemble Classifier
      else:
        print("> 100k")
        self.init_SGDClassifier()
        # Start an SGD Classifier, Kernel Approximation.
    elif self.task == 'regression':
      print('regression')
      num_records = self.df.shape[0]
      if num_records < big_dataset_size:
        print("< 100k")
        self.init_SVR(kernel='linear')
        self.init_SVR(kernel='rbf')
        self.init_RandomForestRegressor()
        # Ridge, linearSVR, rbfSVR, Ensemble Regressors

      else:
        print("> 100k")

  def init_linearSVC(self):
    model = LinearSVC(random_state=0, max_iter=2000)
    model.fit(self.X_train, self.y_train)
    y_pred = model.predict(self.X_test )
    accuracy = accuracy_score(self.y_test, y_pred)
    self.models.append((model, accuracy))

  def init_knn(self, improvement_threshold=0.03, not_improved_number=2):
    best_model = None
    best_accuracy = 0
    not_improved = 0
    i = 1

    while not_improved < not_improved_number:
      not_improved += 1
      if self.verbose == 1:
        print(f"KNN: trying {i} neighbors, currently at acc {best_accuracy}")
      model = KNeighborsClassifier(n_neighbors=i)
      model.fit(self.X_train, self.y_train)
      y_pred = model.predict(self.X_val)
      accuracy = accuracy_score(self.y_val, y_pred)
      if accuracy  > best_accuracy + improvement_threshold:
        best_model = model
        best_accuracy = accuracy
        not_improved = 0
      i+=2

    y_pred = best_model.predict(self.X_test)
    accuracy = accuracy_score(self.y_test, y_pred)
    self.models.append((best_model, accuracy))

  def init_SVC(self):
    model = SVC(gamma='auto', max_iter=2000)
    model.fit(self.X_train, self.y_train)
    y_pred = model.predict(self.X_test)
    accuracy = accuracy_score(self.y_test, y_pred)
    self.models.append((model, accuracy))

  def init_RandomForestClassifier(self, improvement_threshold=0.03, not_improved_number=2):
    best_model = None
    best_accuracy = 0
    not_improved = 0
    i = 1

    while not_improved < not_improved_number:
      not_improved += 1
      if self.verbose:
        print(f"RFC: trying depth {i}, currently  at acc {best_accuracy}")
      model = RandomForestClassifier(max_depth=i, random_state=0)
      model.fit(self.X_train, self.y_train)
      y_pred = model.predict(self.X_val)
      accuracy = accuracy_score(self.y_val, y_pred)
      if accuracy > best_accuracy + improvement_threshold:
        best_model = model
        best_accuracy = accuracy
        not_improved = 0
      i += 2

    y_pred = best_model.predict(self.X_test)
    accuracy = accuracy_score(self.y_test, y_pred)
    self.models.append((model, accuracy))

  def init_SGDClassifier(self):
    model = SGDClassifier(max_iter=2000)
    model.fit(self.X_train, self.y_train)
    y_pred = model.predict(self.X_test)
    accuracy = accuracy_score(self.y_test, y_pred)
    self.models.append((model, accuracy))

  def init_SVR(self, kernel='linear'):
    model = SVR(kernel=kernel)
    model.fit(self.X_train, self.y_train)
    y_pred = model.predict(self.X_test)
    cost = mse(self.y_test, y_pred)
    self.models.append((model, cost))

  def init_RandomForestRegressor(self, improvement_threshold=0.5, not_improved_number=2):
    best_model = None
    best_cost = float('inf')
    not_improved = 0
    i = 1

    while not_improved < not_improved_number:
      not_improved += 1
      if self.verbose:
        print(f"RFR: trying depth {i}, currently  at acc {best_cost}")
      model = RandomForestRegressor(max_depth=i, criterion="squared_error")
      model.fit(self.X_train, self.y_train)
      y_pred = model.predict(self.X_val)
      cost = mse(self.y_val, y_pred)
      if cost + improvement_threshold < best_cost:
        best_model = model
        best_cost = cost
        not_improved = 0
      i += 2

    y_pred = best_model.predict(self.X_test)
    cost = mse(self.y_test, y_pred)
    self.models.append((best_model, cost))
