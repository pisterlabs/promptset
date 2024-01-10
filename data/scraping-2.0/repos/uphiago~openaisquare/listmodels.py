import openai

OPENAI_API_KEY = 'your key here'
openai.api_key = OPENAI_API_KEY


model_lst = openai.Model.list()

for i in model_lst['data']:
    print(i['id'])