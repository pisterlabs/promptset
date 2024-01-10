import os
import openai
import numpy as np
import time

import time
import pandas as pd

MODEL_AUGMENT = "gpt-3.5-turbo-16k"
MODEL_ANSWER = "gpt-3.5-turbo-16k"

openai.api_key = "Paste your openai API key here"

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding(text, model="text-embedding-ada-002"):
    try:
        text = text.replace("\n", " ")
    except:
        None
    try:
        return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
    except:
        time.sleep(2)

def augment_query(query):
    SYS_PROMPT = """
        On [current date: 19 July], you'll receive a finance-related question from a sales manager, without direct interaction. Generate a JSON response with the following structure, considering the temporal aspect:

        {
        "timestamps": # Relevant timestamps to study corresponding tweets for a temporal dynamic aspect (e.g., topic drift). USE THE MINIMAL NUMBER OF TIMESTAMP POSSIBLE ALWAYS ALWAYS!,
        "query": # Repeat the user's query,
        "similarity_boilerplate": # Boilerplate of relevant documents for cosine similarity search after embedding (it could look like example of tweets that might help answer the query),
        }

        Allowed historical timestamps:
        ['2018-07-18', '2018-07-19', '2018-07-08', '2018-07-09', '2018-07-10', '2018-07-11', '2018-07-12', '2018-07-13', '2018-07-14', '2018-07-15', '2018-07-16', '2018-07-17']

        Ensure the output is always in JSON format and never provide any other response.
        """
    response = openai.chat.completions.create(
        model=MODEL_AUGMENT,
        messages=
        [
            {
            "role": "system",
            "content": SYS_PROMPT
             },
            {
            "role": "user",
            "content": query
            }
        ],
        temperature=1,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        ).choices[0].message.content
    return response


def semantic_search(df_loc, query,timestamp, nb_elements_to_consider=15):
    timestamp = str(timestamp).strip()
    embedding = get_embedding(query, model='text-embedding-ada-002')
    filtered_df = df_loc[df_loc["timestamp"]==timestamp].drop(columns=["url"])
    def wrap_cos(x,y):
        try:
            res = cosine_similarity(x,y)
        except:
            res = 0
        return res
    filtered_df['similarity']  = filtered_df.embedding.apply(lambda x: wrap_cos(x, embedding))

    results = filtered_df.sort_values('similarity', ascending=False).head(nb_elements_to_consider)

    return results

def condition_check(tweet, query):
    response = openai.chat.completions.create(model=MODEL_AUGMENT,messages=[    {
        "role": "system",
        "content": "Only answer with True or False no matter what"
        },
        {
        "role": "user",
        "content": f"Consider this tweet:\n\n{tweet}\n\nIs it relevant to the following query:\n\n\{query}"
        }
    ],
    temperature=1,
    max_tokens=1000,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    ).choices[0].message.content
    return bool(response)

def get_number_relevant_tweets(df,timestamp, query):
    sorted_df = semantic_search(df, str(str(query["query"]) + "\n"+  str(query["similarity_boilerplate"])),timestamp, nb_elements_to_consider=len(df))
    left, right = 0, len(sorted_df) - 1
    while left <= right:
        mid = (left + right) // 2
        print(f"Currently searching with max range at {mid}")
        if condition_check(sorted_df['text'].iloc[mid], query):
            left = mid + 1
        else:
            right = mid - 1
    print(f"Dichotomy done, found relevant tweets: {left}")
    return left



def get_relevant_documents(df, query,nb_elements_to_consider = 10):
    query = eval(query)
    all_retrieved = []
    for timestamp in query["timestamps"]:
        number_of_relevant_tweets = get_number_relevant_tweets(df,timestamp, query)
        all_retrieved.append({
            "timestamp" : timestamp,
            "number_of_relevant_tweets": str(number_of_relevant_tweets),
            "tweets" : semantic_search(df, str(str(query["query"]) + "\n"+  str(query["similarity_boilerplate"])),timestamp, nb_elements_to_consider=min(nb_elements_to_consider,number_of_relevant_tweets))
        })
    return all_retrieved

def get_final_answer(relevant_documents, query):
    context = ""
    for document in relevant_documents:
        print("TIMESTAMP: ", document["timestamp"] )
        tweet_entry = document["tweets"]
        context += "\nTimestamp: " + document["timestamp"] + " - Number of relevant tweets in database (EXACT VOLUME OF TWEETS): +"+ document["number_of_relevant_tweets"] + "\nList of tweets:\n" + str((tweet_entry["text"] + "   --- Tweeted by: @" +tweet_entry["source"] +  " \n").to_list()) + "\n---"


    SYS_PROMPT =  f"""
        You will be fed a list of tweets each at a specific timestamp and the number of relevant tweets. You need to take into account (if needed) the number of tweets relevant to the query and how this number evolved. Your task is to use those tweets to answer to the best of your knowledge the following question:

        QUESTION: {query}

        SPECIFIC INSTRUCTIONS AND SYSTEM WARNINGS: You redact a properly structured markdown string containing a professional report.
        You ALWAYS specify your sources by citing them (no urls though). Those tweets are samples from the data and are the closest to the query, you should also take into account the volume of tweets obtained.
        Otherwise, it will be considered highly misleading and harmful content.
        You should however always try your best to answer and you need to study in depth the historical relationship between the timestamps and how it answers the QUESTION.
        You never refer to yourself.
        Make it as if a real human provided a well constructed and structured report/answer extracting the best of the knowledge contained in the context."
        """
    response = openai.chat.completions.create(
                                                model=MODEL_ANSWER,
                                                messages=[
                                                    {
                                                    "role": "system",
                                                    "content": SYS_PROMPT
                                                            },
                                                    {
                                                    "role": "user",
                                                    "content": str(context)
                                                    }
                                                ],

                                                temperature=1,
                                                max_tokens=3000,
                                                top_p=1,
                                                frequency_penalty=0,
                                                presence_penalty=0,
                                                ).choices[0].message.content
    return response

def get_answer(query, df,nb_elements_to_consider=10):
    augmented_query = augment_query(query)

    relevant_documents = get_relevant_documents(df, augmented_query,nb_elements_to_consider=nb_elements_to_consider)

    response = get_final_answer(relevant_documents, augmented_query)
    print(response)


    return response


