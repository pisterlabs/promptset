import openai
import config

openai.api_key = config.openai_api_key
openai.Model.retrieve("gpt-3.5-turbo-instruct")

def openai_predict_response(message):
    response = openai.Completion.create(model="gpt-3.5-turbo-instruct",prompt=message)
    return response["choices"][0]["text"]

def get_embedding(sentence, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=sentence,model=model)
    return response['data'][0]['embedding']