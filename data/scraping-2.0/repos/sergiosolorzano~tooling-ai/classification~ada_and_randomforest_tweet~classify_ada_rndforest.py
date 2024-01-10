#!/usr/bin/env python3

# necessary libraries 
import os
import time
import math
import logging
import datetime
import pathlib
import csv
import ast  # Import ast (Abstract Syntax Trees) module for literal_eval

import openai 
import pandas as pd 
import numpy as np 
# libraries to develop and evaluate a machine learning model 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, accuracy_score 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, accuracy_score 
from sklearn.metrics import confusion_matrix 

openai.api_key = os.environ.get('OPENAI_API_KEY')

current_time = datetime.datetime.now().strftime("%H-%M-%S-%f")[:-3]  # Get time up to milliseconds
log_file_name = f"my_log_file_{current_time}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_file_name,
    filemode='a'
)
logger = logging.getLogger()

#show max col width
pd.set_option('display.max_colwidth', None)

#get embedding function
def get_embedding(text, model="text-embedding-ada-002"):
    return openai.Embedding.create(input = text, model=model)['data'][0]['embedding']

def print_all(dataframe):
    #print all
    # Set display options to show all rows and columns
    pd.set_option('display.max_rows', None)    # Show all rows
    pd.set_option('display.max_columns', None) # Show all columns
    print(df)
    #logger.info(df)

def save_to_csv(embedding_dict):
    dir_path = pathlib.Path.cwd()
    filename = f"embedding_feature_and_class_df_{current_time}.csv"
    filename_path = pathlib.Path(dir_path,filename)
    with open(filename_path,'w',newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['class_embedding','embedding_feature'])
        for key,value in embedding_dict.items():
            row=[key,value]
            csv_writer.writerow(row)

#load csv and ignore encoding errors and skip any bad line
df = pd.read_csv('tweets.csv',encoding_errors="ignore",on_bad_lines='skip')
#print(); print(df.shape)
#logger.info("");logger.info(df.shape)
#drop cols with nulls
df = df.dropna(axis=1)
df = df.rename(columns={'v1': 'OUTPUT'})
df = df.rename(columns={'v2': 'TWEET'})
#print();print(); print(df.head())
#logger.info("");logger.info("");logger.info(df.head())

#get a sample of the population
sample_size=10

#take the first {sample_size} datapoints
#df = df.iloc[:sample_size] 

#or

#take pct X% of spam, rest of ham
# Select 25 'ham' rows and 25 'spam' rows
pct_spam = 0.5
spam_rows = df[df['OUTPUT'] == 'spam'].head(math.ceil(sample_size*pct_spam))
ham_rows = df[df['OUTPUT'] == 'ham'].head(math.ceil(sample_size*(1-pct_spam)))
df = pd.concat([ham_rows, spam_rows])

filtered_df = df[df['OUTPUT'].isin(['ham', 'spam'])]
count_ham = (filtered_df['OUTPUT'] == 'ham').sum()
count_spam = (filtered_df['OUTPUT'] == 'spam').sum()
#print(); print(df.head())
print();print(); print("Sample Size to Create Embeddings:",sample_size)
logger.info("");logger.info("");logger.info(f"Sample Size to Create Embeddings:{sample_size}")
print_all(filtered_df)
print();print(f"Ham:{count_ham} or {count_ham/len(filtered_df)*100:.2f}% - Spam:{count_spam} or {count_spam/len(filtered_df)*100:.2f}%")
logger.info("");logger.info(f"Ham:{count_ham} or {count_ham/len(filtered_df)*100:.2f}% - Spam:{count_spam} or {count_spam/len(filtered_df)*100:.2f}%")

#generate vectors for the dataset 
# #and convert the result to a string array and insert it into the dataset as a column
print(); print("Send Request for Embeddings")
logger.info("");logger.info("Send Request for Embeddings")
#df["embedding_feature"] = df.TWEET.apply(get_embedding).apply(np.array)
df["embedding_feature"] = get_embedding(df['TWEET'].tolist())

#relabel spam and ham
class_dict = {"spam":1,"ham":0}
df["class_embeddings"] = df.OUTPUT.map(class_dict)
#print(df.head())
print(); print("Create Dataframe with Embedding as features and class_embedding as class labels:");print(df.head())
logger.info("");logger.info("Create Dataframe with Embedding as features and class_embedding as class labels:");logger.info(df.head())

#create a dict of the training set embedding & class_embedding column and save to csv
embedding_dataframe = pd.DataFrame(df[['class_embeddings','embedding_feature']])
# Save the DataFrame to a CSV file
print(); print("Save dataframe to CSV.")
logger.info("");logger.info("Save Dataframe to CSV.")
df.to_csv('embedding_dataframe.csv', index=False)
# Read the CSV file back into a DataFrame
df = pd.read_csv('output_file.csv')
df['embedding_feature'] = df['embedding_feature'].str.replace(' ', ',')
df['embedding_feature'] = df['embedding_feature'].apply(ast.literal_eval)
#print_all(retrieved_df)

#divide dataset into training and validation
x = np.array(df.embedding_feature)
y = np.array(df.class_embeddings)
#20% of data for testing. by setting random_state to a fixed value, every time we run this it'll split the same way
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#train random forest classifier with 100 decision trees
print(); print("Start to train the model.")
logger.info("");logger.info("Start to train the model.")
start_time = time.time()
clf = RandomForestClassifier(n_estimators=100)
#learn the relationship between features and class labels
clf.fit(x_train.tolist(),y_train)
end_time = time.time()
elapsed_time_seconds = end_time - start_time
minutes = int(elapsed_time_seconds // 60)
seconds = int(elapsed_time_seconds % 60)
milliseconds = int((elapsed_time_seconds - int(elapsed_time_seconds)) * 1000)
print(f"Time elapsed to train the model for {sample_size} tweets: {minutes} minutes, {seconds} seconds, {milliseconds} milliseconds")
logger.info(f"Time elapsed to train the model for {sample_size} tweets: {minutes} minutes, {seconds} seconds, {milliseconds} milliseconds")

#predict the label
preds = clf.predict(x_test.tolist())

#generate a classification report involving f-1 score, recall, precision, accuracy
report = classification_report(y_test,preds)
print();print(report)
logger.info(""); logger.info(report)
