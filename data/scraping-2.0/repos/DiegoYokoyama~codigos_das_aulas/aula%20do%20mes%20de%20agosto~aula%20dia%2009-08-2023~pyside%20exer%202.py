import openai 

openai.api_key = ''
def get_completion(prompy, model = 'gpt-3-5-turbo'):
    mensagens =[{'role': 'user', 'content': prompy}]
    response = openai.ChatCompletion.create(
        model = model,
        mensages = messages,
        temperatura = 0,
    )
    return response.choises[0].message['content']
prompt = '<Sua pergunta aqui>'
response = get_completion(prompy)
print(response)