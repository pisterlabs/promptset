import openai

api_key = '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'

def send_request(message):
    openai.api_key = api_key
    chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                   messages=[{"role": "user", "content": message}])
    return chat_completion.choices[0].message['content']
