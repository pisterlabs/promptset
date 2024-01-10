import openai
import pandas as pd 

openai.api_key = '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'
# Read all menus
data = pd.read_excel('./recipe.xlsx', usecols = ['음식 제목'])
print(data)

# Chat with ChatGPT 3.5
messages = []

while True:
    for iter in data.itertuples():
        menu = iter[1]
        content=f'Please explain about {menu} in 30 words.'
        messages.append({"role": "user", "content": content})

        completion = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            messages = messages
        )

        f = open("./instruction.txt", 'a')

        chat_response = completion.choices[0].message.content
        chat_response = chat_response.replace(',', '#') + '\n'

        print(f'ChatGPT:{chat_response}')
        messages.append({'role': 'assistant', 'content': chat_response})
        
        f.close()