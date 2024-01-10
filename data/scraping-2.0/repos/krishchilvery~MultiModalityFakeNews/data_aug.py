import pandas as pd # version - 1.3.5
import numpy as np # version - 1.21.6
import matplotlib.pyplot as plt # version - 3.5.3
pd.options.mode.chained_assignment = None  # default='warn'

from tqdm import tqdm
import nltk # version - 3.7
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from transformers import MarianMTModel, MarianTokenizer # transformers version - 4.20.1
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback, AutoModelForCausalLM
from collections import defaultdict

import random
import requests # version - 2.28.1
import json # version - 2.0.9
import time
import openai # version - 0.25.0

# reading all the training, validation, and testing files for the whole data
train_all = pd.read_csv("./all_train.tsv", sep = '\t') # path to be changed based on the path of data file
validate_all = pd.read_csv("./all_validate.tsv", sep = '\t') # path to be changed based on the path of data file
test_all = pd.read_csv("./all_test_public.tsv", sep = '\t') # path to be changed based on the path of data file


# function to remove rows containing NaNs and irrelevant columns
def data_clean(data):
    # total rows in data
    print("Total no. of rows in data:", len(data))
    
    # checking numbers of NaNs in 'clean_title' column
    print("Total no. of NaNs in 'clean_title' column:", data['clean_title'].isnull().sum())
    
    # removing the rows containing NaNs in 'clean_title' column
    data = data[data['clean_title'].notna()]
    print("Total no. of rows in data after removing NaNs:", len(data))
    
    # deleting the irrelevant columns
    data.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'author', 'created_utc', 'domain', 
               'id', 'linked_submission_id', 'num_comments', 'score', 'subreddit', 'title', 'upvote_ratio'], axis=1, inplace=True)
    
    # count of true and false titles
    print("Count of false and true titles:", data['2_way_label'].value_counts())
    print("Ratio of false and true titles:", data['2_way_label'].value_counts(normalize=True))
    
    # avg number of words in title of true and false titles
    data['word_count'] = data['clean_title'].str.split().str.len()
    print("Average count of words in false and true titles:", data.groupby('2_way_label')['word_count'].mean())
    
    return data


# creating the cleaned up training data
train_all = data_clean(train_all)

# partitioning the data into true and false
train_all_true = train_all.loc[train_all['2_way_label'] == 1]
train_all_false = train_all.loc[train_all['2_way_label'] == 0]

# shuffling the datasets
train_all_true = train_all_true.sample(frac=1).reset_index(drop=True)
train_all_true_series = train_all_true['clean_title'].to_list()

train_all_false = train_all_false.sample(frac=1).reset_index(drop=True)
train_all_false_series = train_all_false['clean_title'].to_list()

# selecting samples to be used for augmentation
aug_samples = 10000 # should be a multiple of 50
train_all_true_bt = train_all_true[:aug_samples]['clean_title'].to_list() # data for back translation
train_all_true_gpt = train_all_true[aug_samples: 2*aug_samples]['clean_title'].to_list() # data for GPT-3

train_all_false_bt = train_all_false[:aug_samples]['clean_title'].to_list() # data for back translation
train_all_false_gpt = train_all_false[aug_samples: 2*aug_samples]['clean_title'].to_list() # data for GPT-3


# functions for back translation (Reference - https://amitness.com/back-translation/)
target_model_name = 'Helsinki-NLP/opus-mt-en-fr'
target_tokenizer = MarianTokenizer.from_pretrained(target_model_name)
target_model = MarianMTModel.from_pretrained(target_model_name)

en_model_name = 'Helsinki-NLP/opus-mt-fr-en'
en_tokenizer = MarianTokenizer.from_pretrained(en_model_name)
en_model = MarianMTModel.from_pretrained(en_model_name)


def translate(texts, model, tokenizer, language="fr"):
    # Prepare the text data into appropriate format for the model
    template = lambda text: f"{text}" if language == "en" else f">>{language}<< {text}"
    src_texts = [template(text) for text in texts]

    # Tokenize the texts
    #encoded = tokenizer.prepare_seq2seq_batch(src_texts)
    encoded = tokenizer.prepare_seq2seq_batch(src_texts,return_tensors="pt")
    
    # Generate translation using model
    translated = model.generate(**encoded)

    # Convert the generated tokens indices back into text
    translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
    
    return translated_texts


def back_translate(texts, source_lang="en", target_lang="fr"):
    # Translate from source to target language
    fr_texts = translate(texts, target_model, target_tokenizer, 
                         language=target_lang)

    # Translate from target language back to source language
    back_translated_texts = translate(fr_texts, en_model, en_tokenizer, 
                                      language=source_lang)
    
    return back_translated_texts


# breaking augmentation data into batches of 50 for easier processing
num_batches = int(aug_samples / 50)


# function which takes a list as input and returns a dictionary of lists containing the outputs of back translation
def back_translation_output(data):
    i = 0
    d_data = defaultdict(list)
    d_data_out = defaultdict(list)
    for i in tqdm(range(num_batches)):
        #print(i)
        d_data[i] = data[i*50:(i+1)*50]
        d_data_out[i] = back_translate(d_data[i])
        
    return d_data_out


d_train_all_true_bt_out = back_translation_output(train_all_true_bt)
d_train_all_false_bt_out = back_translation_output(train_all_false_bt)


# functions for GPT 3 data augmentation
# function which takes a list as iput and return a list containing the ouputs of GPT3 augmentation
def gpt3(data):
    new_texts = []
    openai.api_key = "" # to be changed by the user 
    iter_num = 0
    for iter_num in range(50):
        text1 = data[iter_num]
        response = openai.Completion.create(model="text-davinci-002",
                                            prompt=f"Reiterate the given text in a manipulative way.\n{text1}\n\n\n",
                                            temperature=0.5,
                                            max_tokens=30,
                                            top_p=0.95,
                                            frequency_penalty=0,
                                            presence_penalty=0
                                            )
        
        time.sleep(1.5)
        data_new = response['choices'][0]['text'].strip('\n')
        
        if len(data_new) < 2:
            # the format of the response is invalid
            continue

        text = data_new
        new_texts.append(text)
        iter_num += 1
        
    return new_texts


# function which takes a list as input and returns a dictionary of lists containing the outputs of GPT3 augmentation
def gpt3_output(data):
    i = 0
    d_data = defaultdict(list)
    d_data_out = defaultdict(list)
    for i in tqdm(range(num_batches)):
        d_data[i] = data[i*50:(i+1)*50]
        d_data_out[i] = gpt3(d_data[i])
        
    return d_data_out


d_train_all_true_gpt_out = gpt3_output(train_all_true_gpt)
d_train_all_false_gpt_out = gpt3_output(train_all_false_gpt)


# combining the original true examples with the back translation augmented ones
aug_true_bt = []
i = 0
for i in range(len(d_train_all_true_bt_out)):
    aug_true_bt.extend(d_train_all_true_bt_out[i])
    
train_all_true_aug_bt = train_all_true_series
train_all_true_aug_bt.extend(aug_true_bt)
df_train_all_true_aug_bt = pd.DataFrame(train_all_true_aug_bt, columns = ['clean_title'])
df_train_all_true_aug_bt.insert(1, "2_way_label", 1)
df_train_all_true_aug_bt.head()


# combining the original false examples with the back translation augmented ones
aug_false_bt = []
i = 0
for i in range(len(d_train_all_false_bt_out)):
    aug_false_bt.extend(d_train_all_false_bt_out[i])
    
train_all_false_aug_bt = train_all_false_series
train_all_false_aug_bt.extend(aug_false_bt)
df_train_all_false_aug_bt = pd.DataFrame(train_all_false_aug_bt, columns = ['clean_title'])
df_train_all_false_aug_bt.insert(1, "2_way_label", 0)
df_train_all_false_aug_bt.head()


# combining both the true and false back translation datasets into a single one
df_train_all_aug_bt = df_train_all_true_aug_bt.append(df_train_all_false_aug_bt, ignore_index = True)
df_train_all_aug_bt = df_train_all_aug_bt.sample(frac=1).reset_index(drop=True)
df_train_all_aug_bt.to_csv('bt_augmentation.csv')


# combining the original true examples with the GPT 3 augmented ones
aug_true_gpt = []
i = 0
for i in range(len(d_train_all_true_gpt_out)):
    aug_true_gpt.extend(d_train_all_true_gpt_out[i])
    
train_all_true_aug_gpt = train_all_true_series
train_all_true_aug_gpt.extend(aug_true_gpt)
df_train_all_true_aug_gpt = pd.DataFrame(train_all_true_aug_gpt, columns = ['clean_title'])
df_train_all_true_aug_gpt.insert(1, "2_way_label", 1)
df_train_all_true_aug_gpt.head()


# combining the original true examples with the GPT 3 augmented ones
aug_false_gpt = []
i = 0
for i in range(len(d_train_all_false_gpt_out)):
    aug_false_gpt.extend(d_train_all_false_gpt_out[i])
    
train_all_false_aug_gpt = train_all_false_series
train_all_false_aug_gpt.extend(aug_false_gpt)
df_train_all_false_aug_gpt = pd.DataFrame(train_all_false_aug_gpt, columns = ['clean_title'])
df_train_all_false_aug_gpt.insert(1, "2_way_label", 1)
df_train_all_false_aug_gpt.head()


# combining both the true and false GPT 3 datasets into a single one
df_train_all_aug_gpt = df_train_all_true_aug_gpt.append(df_train_all_false_aug_gpt, ignore_index = True)
df_train_all_aug_gpt = df_train_all_aug_gpt.sample(frac=1).reset_index(drop=True)
df_train_all_aug_gpt.to_csv('gpt_augmentation.csv')


# combining the original true examples with the back translation and GPT 3 augmented ones
train_all_true_aug_bt_gpt = train_all_true_series
train_all_true_aug_bt_gpt.extend(aug_true_bt)
train_all_true_aug_bt_gpt.extend(aug_true_gpt)

df_train_all_true_aug_bt_gpt = pd.DataFrame(train_all_true_aug_bt_gpt, columns = ['clean_title'])
df_train_all_true_aug_bt_gpt.insert(1, "2_way_label", 1)
df_train_all_true_aug_bt_gpt.head()


# combining the original false examples with the back translation and GPT 3 augmented ones
train_all_false_aug_bt_gpt = train_all_false_series
train_all_false_aug_bt_gpt.extend(aug_false_bt)
train_all_false_aug_bt_gpt.extend(aug_false_gpt)

df_train_all_false_aug_bt_gpt = pd.DataFrame(train_all_false_aug_bt_gpt, columns = ['clean_title'])
df_train_all_false_aug_bt_gpt.insert(1, "2_way_label", 1)
df_train_all_false_aug_bt_gpt.head()


# combining both the true and false back translation and GPT 3 datasets into a single one
df_train_all_aug_bt_gpt = df_train_all_true_aug_bt_gpt.append(df_train_all_false_aug_bt_gpt, ignore_index = True)
df_train_all_aug_bt_gpt = df_train_all_aug_bt_gpt.sample(frac=1).reset_index(drop=True)
df_train_all_aug_bt_gpt.to_csv('bt_gpt_augmentation.csv')




