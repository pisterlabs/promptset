
import sys 
import json


import openai
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer


from openai_interface.prompt_gpt3 import prompt_gpt3_test, gpt3_embeddings, prompt_gpt3
from metrics.distance_metrics import *
from metrics.other_metrics import *
from utils.utils import * 
from csv_manipulation.csv_utils import add_column, check_columns

import openai_interface.prompt_gpt3 as openai_interface
import piazza_api_utils.utils as piazza_utils



scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3', 'rougeL'], use_stemmer=False)

def usage():
    print('Usage: python compute_metrics.py <csv-path>')



def fscore(precision, recall):
    return 2 * ((precision * recall) / (precision + recall))

def precision(tp, fp):

    return tp / (tp + fp)

def recall(tp, fn):
    return tp / (tp+fn)

def accuracy(tp, tn, fp, fn):
    return (tn + tp) / (tp + tn + fp + fn)



def find_max_rouge_score(question:str, series:pd.Series):
    """Find and return answer in series which has the highest rogue score with the question"""
    max_index = -1
    max_fscore = -1
    for index, value in series.items():
        if type(value) == float and np.isnan(value):
            continue 
        else:

            scores = scorer.score(question, value)
            # print(scores['rouge1'].fmeasure)
            
            if scores['rouge1'].fmeasure > max_fscore:
                max_index = index
                max_fscore = scores['rouge1'].fmeasure

    return max_index, max_fscore

def pairwise_rogue_score(text1, text2):
    scores = scorer.score(text1, text2)
    return scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, scores['rougeL'].fmeasure


def pairwise_distance_metric(e1, e2, distance_metric='cosine_similarity'):
    if distance_metric == 'cosine_similarity':
        return my_cosine_similarity(e1, e2)
    else:
        return my_euclidean_distance(e1, e2)


def compute_embedding_distance(row):
    """ Compute distance metrics between embedding vectors in the df row

        instructor answer vs. model answer (single answer)
        instructor answer vs. answer1, answer2 (double answer)
    """
    if not is_empty(row['instructor_answer']):
        instructor_embedding_cohere = json.loads(row['instructor_answer_embeddings_cohere'])
        model_embedding_cohere = json.loads(row['model_answer_embeddings_cohere'])

        instructor_embedding_openai = json.loads(row['instructor_answer_embeddings_openai'])
        model_embedding_openai = json.loads(row['model_answer_embeddings_openai'])
        
        row['cohere_cosine_distance'] =  pairwise_distance_metric(instructor_embedding_cohere, model_embedding_cohere)
        row['openai_cosine_distance'] =  pairwise_distance_metric(instructor_embedding_openai, model_embedding_openai)

        row['cohere_euclidean_distance'] =  pairwise_distance_metric(instructor_embedding_cohere,  model_embedding_cohere, distance_metric='e')
        row['openai_euclidean_distance'] =  pairwise_distance_metric(instructor_embedding_openai, model_embedding_openai, distance_metric='e' )
    
    return row



def compute_rouge_score(row):

    if not is_empty(row['instructor_answer']):
        
        instructor_answer = row['instructor_answer'] 
        model_answer = row['model_answer']
        
        rouge1, rouge2, rougeL = pairwise_rogue_score(instructor_answer, model_answer)

        row['rouge1'] = rouge1
        row['rouge2'] = rouge2
        row['rougeL'] = rougeL

    return row 


def add_perplexity(row, rowname='instructor_answer_logprobs'):

    if not is_empty(row[rowname]):
        logprobs = json.loads(row[rowname])
        perplexity = compute_perplexity(logprobs)
        row['instructor_answer_perplexity'] = perplexity
        row['instructor_answer_logprobs_avg'] = np.mean(logprobs)
    return row



def compute_aggregate_statistics(df:pd.DataFrame, groupby_column='classification', 
                                 columns=['openai_cosine_distance', 'openai_euclidean_distance', 'cohere_cosine_distance', 'cohere_euclidean_distance',
                                          'rouge1', 'rouge2', 'rougeL', 'instructor_answer_perplexity', 'instructor_answer_logprobs_avg']):
    
    
    if not groupby_column:
        return df.mean(), df.std() 
    
    columns.append(groupby_column)
    df = df[columns]

    return df.groupby(groupby_column).mean(), df.groupby(groupby_column).std() 


def compute_metrics_on_csv(args, df):

    if args.metric == 'all':
        answer_text_columns = ["instructor_answer", "model_answer"]
        embedding_columns = ["instructor_answer_embeddings_cohere", "instructor_answer_embeddings_openai", 
                            "model_answer_embeddings_cohere", "model_answer_embeddings_cohere"]

        logprobs_columns = ["instructor_answer_logprobs"]

        check_columns(answer_text_columns + embedding_columns + logprobs_columns, df)
      
        df = df.apply(lambda row: compute_embedding_distance(row), axis=1)
        df = df.apply(lambda row: compute_rouge_score(row), axis=1)
        df = df.apply(lambda row: add_perplexity(row), axis=1)

    if args.metric == 'rouge':
        df = df.apply(lambda row: compute_rouge_score(row), axis=1)

    if args.metric == 'embedding_distance':
        df = df.apply(lambda row: compute_embedding_distance(row), axis=1)

    if args.metric == 'perplexity':
        df = df.apply(lambda row: add_perplexity(row), axis=1)

        
    return df



def main():
    if len(sys.argv) != 2:
        usage()
        exit(-1)
    
    df:pd.DataFrame = pd.read_csv(sys.argv[1])
    if check_columns():
        df = df.apply(lambda row: compute_embedding_distance(row), axis=1)
        df = df.apply(lambda row: compute_rouge_score(row), axis=1)
        df = df.apply(lambda row: add_perplexity(row), axis=1)

    return df

    

if __name__ == '__main__':
    main()