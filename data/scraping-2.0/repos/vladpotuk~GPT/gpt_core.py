import openai

api_key = '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'

def send_request(message):
    openai.api_key = api_key
    chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                   messages=[{"role": "user", "content": message}])
    return chat_completion.choices[0].message['content']
