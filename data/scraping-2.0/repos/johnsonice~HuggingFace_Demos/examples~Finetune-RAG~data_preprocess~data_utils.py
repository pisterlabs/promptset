## data process utils

#%%
import openai
import os, uuid
from functools import wraps
import logging
#from sentence_transformers.evaluation import InformationRetrievalEvaluator

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
#%%