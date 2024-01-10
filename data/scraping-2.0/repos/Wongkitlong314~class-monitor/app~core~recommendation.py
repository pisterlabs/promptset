#%%
import openai
import time
import pandas as pd
import random
import numpy as np
import re
import os
import tempfile
import pandas as pd
import json

openai.api_key = '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'
#获取当前路径
raw_path = os.path.abspath(os.path.dirname(__file__))

#%%
def get_completion(prompt, sys_prompt, model="gpt-3.5-turbo", temperature=0):
    messages = [{"role":"user", "content" : prompt}, {"role":"system", "content" : sys_prompt}]
    response = ''
    except_waiting_time = 0.1
    while response == '':
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                request_timeout=50
            )
            # k_tokens = response["usage"]["total_tokens"]/1000
            # p_tokens = response["usage"]["prompt_tokens"]/1000
            # r_tokens = response["usage"]["completion_tokens"]/1000
            # print("Tokens used: {:.2f}k".format(k_tokens))
            # print("Prompt tokens: {:.2f}k".format(p_tokens))
            # print("Response tokens: {:.2f}k".format(r_tokens))

        except Exception as e:
            print(e)
            print("Sleep for {:.2f}s".format(except_waiting_time))
            time.sleep(except_waiting_time)
            if except_waiting_time < 2:
                except_waiting_time *= 2
    return response.choices[0].message["content"]


#%%
def recommend_books(user_message):
    book_path = raw_path + '/Books/random_books.csv'
    books = pd.read_csv(book_path)
    books['index'] = np.arange(1,len(books)+1)
    random_books = random.sample(range(1,len(books)+1),40)
    random_books = [i for i in random_books]
    random_books = books[books['index'].isin(random_books)]
    candidate_set = random_books['bookTitle'].tolist()


    system_prompt = "I want you to act as a book recommender. My first request is 'I need help finding 5 books from the candidate set that a user want to read given his/her message.' Respond directly to the user using 'you'"
    prompt = (
            f"\nUser Message: {user_message}"
            +f"\nCandidate Set: {candidate_set}"
            +"\nPlease respond with the format: "
            +"\n[book name 1] [brief advertisement and introduction 1]"
            +"\n[book name 2] [brief advertisement and introduction 2] ......"
    )

    response = get_completion(prompt=prompt, sys_prompt=system_prompt, model="gpt-3.5-turbo", temperature=0.2)
    return response


def recommend_movies(user_message):
    movie_path = raw_path + '/Movie/raw_data/movies.dat'
    movies = pd.read_table(movie_path, encoding='ISO-8859-1', sep='::', header=None, names=['movie_id', 'title', 'genres'], engine='python')

    #随机选取40部电影
    random.seed(1)
    random_movies = random.sample(range(1,len(movies)),40)
    random_movies = [i for i in random_movies]
    random_movies = movies[movies['movie_id'].isin(random_movies)]
    candidate_set = random_movies['title'].tolist()


    system_prompt = "I want you to act as a movie recommender. My first request is 'I need help finding 5 movie from the candidate set that a user want to watch given his/her message.' Respond directly to the user using 'you'"
    prompt = (
            f"\nUser Message: {user_message}"
            +f"\nCandidate Set: {candidate_set}"
            +"\nPlease respond with the format: "
            +"\n[movie name 1] [brief advertisement and introduction 1]"
            +"\n[movie name 2] [brief advertisement and introduction 2] ......"
    )

    response = get_completion(prompt=prompt, sys_prompt=system_prompt, model="gpt-3.5-turbo", temperature=0.2)
    return response

#%%

def recommend_news(user_message):
    temp_dir = os.path.join(tempfile.gettempdir(), 'mind')
    os.makedirs(temp_dir, exist_ok=True)
    news_path = os.path.join(temp_dir, 'news.tsv')
    news = pd.read_table(news_path,
                header=None,
                names=[
                    'id', 'category', 'subcategory', 'title', 'abstract', 'url',
                    'title_entities', 'abstract_entities'
                ])
    news['index'] = np.arange(1,len(news)+1)

    random.seed(1)
    random_news = random.sample(range(1,len(news)),20)
    random_news = news[news['index'].isin(random_news)]
    res = ""
    dic = {}
    for i in range(len(random_news)):
        if random_news.iloc[i]['abstract'] != np.nan:
            index = random_news.iloc[i]['index']
            res += f'{index}.'+random_news.iloc[i]['title']+'\n'
            dic[index] = [random_news.iloc[i]['title'],random_news.iloc[i]['abstract']]

    system_prompt = "I want you to act as a news recommender. My first request is 'I need help finding 5 news from the candidate set that a user want to read given his/her message.' The each line in candidate set combines of news id and news title. Respond directly to the user using 'you'"
    prompt = (
            f"\nUser Message: '{user_message}'"
            +f"\nCandidate Set: {res}"
            +"\nPlease respond with the format: "
            +"\n[news id 1]::[news title 1]"
            +"\n[news id 2]::[news title 2] ......"
    )

    response = get_completion(prompt=prompt, sys_prompt=system_prompt, model="gpt-3.5-turbo", temperature=0.2)
    return response
    # print(dic[13760])

    # generate_sys_prompt = "I want you to act as a news content generator. My first request is 'I need help generating news content for at least 300 words given news title and news content.'"
    # generate_prompt = (f"News title: {dic[13760][0]}\n"
    #                    +f"News abstract: {dic[13760][1]}\n"
    #                    )

    # response_2 = get_completion(prompt=generate_prompt, sys_prompt=generate_sys_prompt, model="gpt-3.5-turbo", temperature=0.2)
    # print(response_2)


    # %%

functions = [
        {   
            "id":0,
            "name": "recommend_books",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_message": {
                        "type": "string",
                        "description": "The message that the user sends to the system",
                    },

                    # "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                # "required": ["location"],
            },
            "description":"A function that recommend books to users."
        },
        {
            "id":1,
            "name": "recommend_movies",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_message": {
                        "type": "string",
                        "description": "The message that the user sends to the system",
                    },
                },
            },
            "description":"A function that recommend movies to users."   
        },
        {
            "id":2,
            "name": "recommend_news",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_message": {
                        "type": "string",
                        "description": "The message that the user sends to the system",
                    },
                },
            },
            "description":"A function that recommend news to users."

        }
]


def recommender_filter(functions, user_message):
    # Step 1: send the conversation and available functions to GPT
    messages = [{"role": "user", "content": user_message}]
    functions = functions
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call="auto",  # auto is default, but we'll be explicit
    )
    response_message = response["choices"][0]["message"]
    print(response_message)

    # Step 2: check if GPT wanted to call a function
    if response_message.get("function_call"):
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "recommend_books": recommend_books,
            "recommend_movies": recommend_movies,
            "recommend_news": recommend_news,
        }  # only one function in this example, but you can have multiple
        function_name = response_message["function_call"]["name"]
        function_to_call = available_functions[function_name]
        #print(function_to_call)
        function_args = json.loads(response_message["function_call"]["arguments"])
        function_response = function_to_call(
            user_message=user_message,
            # userId="userId"
        )
        print(function_response)
        return function_response
    else:
        print(response_message["content"])
        return response_message["content"]

        # # Step 4: send the info on the function call and function response to GPT
        # messages.append(response_message)  # extend conversation with assistant's reply
        # messages.append(
        #     {
        #         "role": "function",
        #         "name": function_name,
        #         "content": function_response,
        #     }
        # )  # extend conversation with function response
        # second_response = openai.ChatCompletion.create(
        #     model="gpt-3.5-turbo-0613",
        #     messages=messages,
        # )  # get a new response from GPT where it can see the function response
    

user_message = "i want to read some news?"
recommender_filter(functions, user_message)