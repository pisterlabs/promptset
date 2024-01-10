import os
from flask import Blueprint, jsonify
from googlesearch import search
from langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models.openai import ChatOpenAI
import time
import openai
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
# sys.path.append(os.path.abspath("C:\\Users\\subhr\\School_projects\\ArguMentor\\debate-ai\\flask-server\\backend"))

# Initialize a Flask Blueprint
second = Blueprint("second", __name__, static_folder="static", template_folder="template")

# Set your API keys
os.environ["OPENAI_API_KEY"] = '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'
os.environ["SERPAPI_API_KEY"] = 'ebfaafb043613442e0010e3795c9ead4cab196e5448a6e3728d64edbbccdf731'


# Move the url_list declaration inside the search_google function
def search_google(query, num_results):
    url_list = []
    result_list = search(query, num_results=(num_results+1))

    rec_counter = 0
    for result in result_list:
        if rec_counter < num_results:
            if result not in url_list:
                url_list.append(result)
                rec_counter = rec_counter + 1
                print(result)
        else:
            break
    print (url_list)
    return url_list


# Modify the 'prompt' inside the web_qa function
def web_qa(url_list, query):
    # openai = ChatOpenAI(
    #     model="gpt-3.5-turbo",
    #     temperature=0.7,
    #     max_tokens=2048
    # )
    #
    # messages = [{"role": "user", "content": prompt}]
    #
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=messages,
    #     temperature=0,
    # )
    #
    # return response.choices[0].message["content"]

    results = []  # Create a list to store results
    print(len(url_list))

    for url in url_list:
        print(url)
        messages = [{"role": "user", "content": url}]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0,
        )
        results.append({'url': url, 'answer': response.choices[0].message["content"]})

    print("Before returning from web_qa")
    print(results)
    return results



# def get_completion(prompt, model="gpt-3.5-turbo"):
#
# messages = [{"role": "user", "content": prompt}]
#
# response = openai.ChatCompletion.create(
# model="gpt-3.5-turbo",
# messages=messages,
# temperature=0,
# )
#
# return response.choices[0].message["content"]