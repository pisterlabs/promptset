## data process utils

#%%
import openai
import os, uuid
import pandas as pd
from functools import wraps
import logging
from datasets import Dataset
from sentence_transformers.evaluation import InformationRetrievalEvaluator

openai.api_key  = os.getenv('OPENAI_API_KEY')

def get_completion(prompt, sys_msg = None, model="gpt-3.5-turbo",temperature=0):
    default_sys = "You are an ecumenist at the International Monetary Fund (IMF). Your primary role is grounded in economics. Your main task is to assist the user by providing insights, analyses, and guidance from an economist's perspective. Ensure your responses align with the expertise and perspective of an IMF economist."
    if not sys_msg:
        sys_msg = default_sys
    messages = [{"role": "system", 
                    "content":sys_msg},
                {"role": "user", 
                    "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def construct_retrieve_evaluator(df_eval,q_k='question',c_k='context'):
    """
    df_eval : pandas DF with questions and context as key
    """
    queries = {}
    relevant_docs = {}
    corpus = {}
    context_groups = df_eval.groupby(c_k)
    for group_name,group_df in context_groups:
        corpus_id = str(uuid.uuid4())
        corpus[corpus_id] = group_df[c_k].iloc[0]
        for index,row in group_df.iterrows():
            question_id = str(uuid.uuid4())
            queries[question_id] = row[q_k]
            relevant_docs[question_id] = [corpus_id]
    
    evaluator = InformationRetrievalEvaluator(queries, corpus, relevant_docs)
    
    return evaluator

def load_train_eval_dataset(train_file_path=None,eval_file_path=None):
    keep_cols=['question', 'context', 'answer']
    if train_file_path:
        t_df = pd.read_excel(train_file_path)[keep_cols]
        t_dataset = Dataset.from_pandas(t_df)
    else:
        t_dataset=None
        
    if eval_file_path:
        e_df = pd.read_excel(eval_file_path)[keep_cols]
        e_dataset = Dataset.from_pandas(e_df)
    else:
        e_dataset=None  
          
    return t_dataset,e_dataset

def tokenize_retrieve_ds(input_ds,text_columns_dict,tokenizer):
    for k,v in text_columns_dict.items():
        input_ds = input_ds.map(
                        lambda x: tokenizer(
                                x[v], truncation=True#,return_tensors='pt' #padding='max_length', #max_length=128,
                        ), 
                        batched=True,
                    )
        input_ds = input_ds.rename_column('input_ids', '{}_ids'.format(k))
        input_ds = input_ds.rename_column('attention_mask', '{}_mask'.format(k))
        
    return input_ds
#%%