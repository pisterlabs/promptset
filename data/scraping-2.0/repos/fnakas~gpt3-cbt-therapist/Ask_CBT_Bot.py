import openai
import os
import numpy as np
import pandas as pd
import argparse
from openai.embeddings_utils import get_embedding, cosine_similarity



parser = argparse.ArgumentParser()

parser.add_argument("--key")

parser.add_argument("--question")

args = parser.parse_args()



OPENAI_API_KEY = args.key
openai.api_key = OPENAI_API_KEY
completion = openai.Completion()

def find_similar_doc(docs_embeddings_csv, user_question, n = 5, engine = 'text-search-babbage-query-001'):

    df = pd.read_csv(docs_embeddings_csv)
    df['babbage_similarity'] = df.babbage_similarity.apply(eval).apply(np.array)
    df['babbage_search'] = df.babbage_search.apply(eval).apply(np.array)

    def get_embedding(text, engine="text-similarity-babbage-001"):
       text = text.replace("\n", " ")
       return openai.Embedding.create(input = [text], engine=engine)['data'][0]['embedding']


    def search_reviews(df, user_question, n, pprint=True):
       embedding = get_embedding(user_question, engine=engine)
       df['similarities'] = df.babbage_search.apply(lambda x: cosine_similarity(x, embedding))
       res = df.sort_values('similarities', ascending=False).head(n)
       return res

    res = search_reviews(df, user_question, n)
    
    return(res)



def generate_generic_response(user_question):
    
    prompt_text = user_question + "\n\n###\n\n Hello!"

    response = completion.create(
        engine="text-davinci-002",
        prompt = prompt_text,
        max_tokens = 500,
        temperature=0.3,
        frequency_penalty=0.7,
        presence_penalty=0.3,
        n=1,
        stop=[" END"],
    )

    return(response)

def generate_response(prompt):
    

    response = completion.create(
        engine="text-davinci-002",
        prompt = prompt,
        max_tokens = 1100,
        temperature=0.5,
        frequency_penalty=0.4,
        presence_penalty=0.1,
        n=1,
        stop=[" END"],
    )

    return(response)

def get_cbt_prompt(user_question, gen_response):
    
    example = "User: " + user_question + "   Dr Johnson: Hello!" + gen_response + " END"

    distortion_docs = find_similar_doc("distortion_embeddings.csv", example, n = 9, engine = 'text-search-babbage-query-001')

    distortion_text = distortion_docs.iloc[0]["Description"]
    distortion_name = distortion_docs.iloc[0]["Name"]

    treatment_docs = find_similar_doc("treatments_embeddings.csv", example, n = 14, engine = 'text-search-babbage-query-001')

    treatment_text = treatment_docs.iloc[0]["Description"]
    treatment_name = treatment_docs.iloc[0]["Name"]

    start_distortion = '"' + distortion_name + ' is a CBT cognitive distortion.' + distortion_text + '"' + '  '

    start_treatment = '"' + treatment_name + ' is a CBT treatment.' + treatment_text + '"' + '  '

    start = start_distortion + start_treatment

    middle = "The following is a dialogue between a client and their psychotherapist :" + '  ' + example + '  '

    end = "The following is an alternative response by Dr Weathers, a therapist who specializes in CBT. Dr Weathers first gives a CBT version of Dr Johnson's diagnosis and by explaining why the patient might be suffering from the cognitive distortion called " + distortion_name + ". Finally, Dr Weathers recommends a CBT treatment called " + treatment_name + " and explains how it will help with the client's particular problem by battling " + distortion_name + ". Dr Weathers gives a long and detailed response:  "  + "Dr Weathers:"

    prompt = start + middle + end
    
    return(prompt, distortion_docs, treatment_docs)


def ask_CBT_bot(user_question):

    res = generate_generic_response(user_question)

    gen_response = res["choices"][0]["text"]
    
    prompt, distortion_docs, treatment_docs = get_cbt_prompt(user_question, gen_response)

    out = generate_response(prompt)

    res = out["choices"][0]["text"]

    return(res, gen_response, prompt, distortion_docs, treatment_docs)


print(ask_CBT_bot(args.question)[0])

