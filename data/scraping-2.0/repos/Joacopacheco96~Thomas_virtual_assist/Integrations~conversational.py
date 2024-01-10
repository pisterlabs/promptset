# import os
# import openai
import requests
import json

def conversational(query) :
    url = 'https://www.botlibre.com/rest/json/chat'
    headers = {'Content-Type': 'application/json'}
    data = {
        "application": "6627987816708691542",
        "instance": "165",
        "message": query
    }
    response = requests.post(url, json=data, headers=headers)
    print(response)
    response_text = response.text
    data_dict = json.loads(response_text)
    if data_dict:
        return(data_dict)
    else:
        return ' I cant find a solution for this at the moment'  

    # Print or use the response text as needed




# # openai.organization = 'org-Om0k9Kku79ZnUhQdFl8AVNLP'
# openai.api_key = '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'
# response = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages = [{'role':'user','content':'2 + 2'}],
#     temperature=0
#     )
# print(response)


# from perplexity_api import PerplexityAPI, TimeoutException
# ppl = PerplexityAPI()

# query = "hello world in python"

# try:
#     print(ppl.query(query, follow_up=True))
# except TimeoutException:
#     print("Query timed out:", query)
