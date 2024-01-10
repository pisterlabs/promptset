import openai
import time

import time
import numpy as np

GPT_MODEL_ANSWER = "gpt-3.5-turbo-16k"

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

def format_query(query):

    resp = {
            "timestamps": [],
            "query": query
            }

    return resp

def semantic_search(df_loc, query, nb_programs_to_display=15):

    embedding = get_embedding(query, model='text-embedding-ada-002')
    filtered_df = df_loc.drop(columns=["url"])
    def wrap_cos(x,y):
        try:
            res = cosine_similarity(x,y)
        except:
            res = 0
        return res
    filtered_df['similarity']  = filtered_df.embedding.apply(lambda x: wrap_cos(x, embedding))
    results = filtered_df.sort_values('similarity', ascending=False).head(nb_programs_to_display)
    return results

def get_relevant_documents(df, query, nb_programs_to_display=15):
    all_retrieved= [{
        "timestamp" : "",
        "tweets" : semantic_search(df, query["query"], nb_programs_to_display=nb_programs_to_display)
    }]
    return all_retrieved

def get_final_answer(relevant_documents, query):
    response = relevant_documents[0]
    tweet_entry = response["tweets"]
    context = "\nList of tweets:\n" + str((tweet_entry["text"] + "   --- Tweeted by: @" +tweet_entry["source"] +  " \n").to_list()) + "\n---"
    USER_PROMPT = f"""
    "We have provided context information below. 
    ---------------------
    {context}
    "\n---------------------\n"
    Given the information above, please answer the question: {query}
    """
    
    response = openai.chat.completions.create(
                                            model=GPT_MODEL_ANSWER,
                                            messages=[
                                                {
                                                "role": "user",
                                                "content": USER_PROMPT
                                                }
                                            ],

                                            temperature=1,
                                            max_tokens=1000,
                                            top_p=1,
                                            frequency_penalty=0,
                                            presence_penalty=0,
                                            ).choices[0].message.content
    return response

def get_answer(query, df, api_key):
    """This approach is considered naive because it doesn't augment the user query.
    This means that we try to retrieve documents directly relevant to the user query and then combine them into an answer.
    The query is formatted to have the same structure given to the LLM as the other two approaches

    Args:
        query (String): Query given by the user
        df (pd.DataFrame()): corpus with embeddings
        api_key (String): OpenAI API key

    Returns:
        String: Answer to the original query
    """
    openai.api_key = api_key
    formatted_query = format_query(query)
    relevant_documents = get_relevant_documents(df, formatted_query,nb_programs_to_display=15)
    response = get_final_answer(relevant_documents, formatted_query)
    return response
