import openai

openai.api_key = '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'

def chat_with_gpt3(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response['choices'][0]['message']['content']
