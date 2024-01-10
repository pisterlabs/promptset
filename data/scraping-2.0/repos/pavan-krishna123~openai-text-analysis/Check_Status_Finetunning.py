import openai

# Replace with your API key
with open('openaiapikey.txt', 'r') as infile:
    open_ai_api_key = infile.read()
openai.api_key = open_ai_api_key

def finetune_get(ftid):
    resp = openai.FineTune.retrieve(ftid)
    print(resp)

# Usage example
finetune_id = "ft-zTmRQq66suOeMw3sJrFJUNZS"
finetune_get(finetune_id)
