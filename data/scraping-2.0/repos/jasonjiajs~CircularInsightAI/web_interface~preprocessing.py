from openai import OpenAI
import pandas as pd
import numpy as np
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pickle
import streamlit as st

def read_data(filepath, nrows_to_keep=None):
    if nrows_to_keep is not None:
        df_full = pd.read_csv(filepath, encoding='ISO-8859-1').iloc[:nrows_to_keep, :]
    else:
        df_full = pd.read_csv(filepath, encoding='ISO-8859-1')

    df = df_full[['problem', 'solution']]
    return df_full, df

def get_category(df):
    # Load the kmeans model
    with open('web_interface/kmeans/kmeans.pkl', 'rb') as file:
        kmeans = pickle.load(file)

    with open('web_interface/kmeans/vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    
    # Pre-process data
    df_problem_statements = df['problem'].dropna().values

    # Vectorize problems using the same vectorizer
    df_problem_statements = vectorizer.transform(df_problem_statements)

    # Predict the cluster labels for the test data
    clusters = kmeans.predict(df_problem_statements)

    # Apply mapping to the cluster labels (0,1,2,3,4)
    cluster_mapping = {
    0: "Other/General Waste",
    1: "Plastic Waste",
    2: "Clothing/Textile Waste",
    3: "E-waste",
    4: "Food Waste"
    }
    
    df['category'] = np.array([cluster_mapping[cluster] for cluster in clusters])

    return df

def get_response(client, system_content, user_content, finetuned=False):
    if finetuned:
        model = 'ft:gpt-3.5-turbo-1106:personal::8e9YXb9p'
    else:
        model = "gpt-3.5-turbo-1106"
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
            ],
        temperature=0,
        seed=0
    )
    return json.loads(response.choices[0].message.content)

def get_metrics(df_dict, client, system_content, finetuned):
    response_list = []
    for i in range(len(df_dict)):
        user_content = str(df_dict[i])
        response = get_response(client, system_content, user_content, finetuned)
        response_list.append(response)

    df_metrics = pd.DataFrame(response_list)
    return df_metrics

def get_metrics_for_filtering_ideas(df_full, df, client, finetuned=True):
    df_dict = df.to_dict(orient='records') # Process df into list of dictionaries
    system_content = "You are a venture capital expert evaluating potential circular economy startup pitches. \
    Mark the startup idea (problem and solution) \
    from 1 to 3 in integer numbers (where 1 is bad, 2 is okay, and 3 is good) \
    in each of four criteria: \
    relevance of the problem to the circular economy (relevance_problem), \
    clarity of the problem (clarity_problem), \
    suitability of solution to the problem (suitability_solution) and \
    clarity of the solution (clarity_solution). \
    Return the following fields in a JSON dict: \
    'relevance_problem', 'clarity_problem', 'suitability_solution' and 'clarity_solution'."
    df_metrics = get_metrics(df_dict, client, system_content, finetuned)
    df_metrics = pd.concat([df_full, df_metrics], axis=1)
    df_metrics['filter_score'] = df_metrics[['relevance_problem', 'clarity_problem', 'suitability_solution', 'clarity_solution']].mean(axis=1)
    return df_metrics

def get_metrics_for_ranking_ideas(df_full, df, client, finetuned=False):
    df_dict = df.to_dict(orient='records') # Process df into list of dictionaries
    system_content = "You are a venture capital expert evaluating potential circular economy startup pitches. \
    Evaluate the following problems and solutions using the following metrics: \
    market potential, feasibility, scalability, technological innovation, \
    alignment with circular economy principles and novelty. \
    For each metric return a score from 1 to 5, where 1 is very bad and 5 is very good. \
    Also, for each metric, provide a sentence explaining why you gave the evaluation. \
    Finally, for each metric, provide a sentence giving advice on how the idea can be improved. \
    Here are the criteria: \
    ------ \
    Market Potential: \
    1: Very limited market potential, little to no demand. \
    2: Limited market potential, niche demand. \
    3: Moderate market potential, some demand. \
    4: High market potential, substantial demand. \
    5: Exceptional market potential, widespread demand. \
    --- \
    Feasibility: \
    1: Highly impractical and challenging to implement. \
    2: Feasibility concerns, significant challenges. \
    3: Moderately feasible, with some challenges. \
    4: Feasible, with manageable challenges. \
    5: Highly feasible, minimal obstacles. \
    --- \
    Scalability: \
    1: Limited scalability, difficult to expand. \
    2: Limited scalability, but potential for some expansion. \
    3: Moderate scalability, with reasonable expansion potential. \
    4: Highly scalable, with significant expansion opportunities. \
    5: Exceptionally scalable, easy to expand on a large scale. \
    --- \
    Technological Innovation: \
    1: Outdated and lacks innovation. \
    2: Limited technological innovation, behind industry standards. \
    3: Moderate technological innovation, in line with current standards. \
    4: Advanced technological innovation, ahead of industry standards. \
    5: Cutting-edge technological innovation, sets new industry standards. \
    --- \
    Circular Economy Principles: \
    1: Lacks consideration for circular economy principles. \
    2: Limited integration of circular economy principles. \
    3: Some adherence to circular economy principles. \
    4: Strong incorporation of circular economy principles. \
    5: Fully aligned with and actively promotes circular economy principles. \
    --- \
    Novelty: \
    1: Highly common and lacks uniqueness. \
    2: Limited novelty, similar concepts exist. \
    3: Moderately novel, some unique aspects. \
    4: Highly novel, stands out in the market. \
    5: Extremely novel, groundbreaking and highly unique. \
    ------ \
    Return the scores as the following fields in a JSON dict: \
    market_potential, feasibility, scalability, innovation, alignment, novelty, \
    market_potential_eval, feasibility_eval, scalability_eval, innovation_eval, alignment_eval, novelty_eval, \
    market_potential_advice, feasibility_advice, scalability_advice, innovation_advice, alignment_advice, novelty_advice. \
    Don't be afraid to give low scores and give direct feedback. \
    "
    df_metrics = get_metrics(df_dict, client, system_content, finetuned)
    df_metrics = pd.concat([df_full.reset_index(drop=True), df_metrics.reset_index(drop=True)], axis=1)
    df_metrics['overall_score'] = df_metrics[['market_potential', 'feasibility', 'scalability', 'innovation', 'alignment', 'novelty']].mean(axis=1)
    return df_metrics

