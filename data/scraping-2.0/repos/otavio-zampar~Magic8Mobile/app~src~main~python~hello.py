def helloworld():
    return "eu estou colocando uma string de dentro de python"
    
def callAPI():
    import os
    import openai
    openai.organization = "org-TnOjpyRnuxFqkah99zlgVK87"
    openai.api_key = os.getenv("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")
    openai.Model.list()
    
def makeRequest():
    import requests
    key = "<APIKEY>"
    headers = {"Authorization": f"Bearer {key}"}
    data = {'model': 'text-davinci-002', 'prompt': 'Once upon a time'}
    requests.post(url, headers=headers, json=data).json()
{'id': 'cmpl-6HIPWd1eDo6veh3FkTRsv9aJyezBv', 'object': 'text_completion', 'created': 1669580366, 'model': 'text-davinci-002', 'choices': [{'text': ' there was a castle up in space. In this castle there was a queen who', 'index': 0, 'logprobs': None, 'finish_reason': 'length'}], 'usage': {'prompt_tokens': 4, 'completion_tokens': 16, 'total_tokens': 20}}
