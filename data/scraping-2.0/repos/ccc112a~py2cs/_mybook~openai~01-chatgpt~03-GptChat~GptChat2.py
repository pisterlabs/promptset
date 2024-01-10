import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

opList = ['quit', 'question', 'q', 'file', 'f']

user = input('user:')

while True:
    command = input('command:').lower()
    args = command.split()
    op = args[0]
    if op not in opList:
        print(f'Oparation error, only {opList} allowed!')
        continue
    if op == 'quit': break
    if op == 'q' or op == 'question':
        question = ' '.join(args[1:])
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a chatbot"},
                {"role": "user", "content": f"{question}"}
            ]
        )  
        print(response['choices'][0]['message']['content'])
    if op == 'f' or op == 'file':
        if response is None:
            print('errro: no response to save!')
        else:
            mode = args[1]
            fname = args[2]
            with open(fname, mode, encoding='utf-8') as file:
                file.write(f"## {user}: {question}\n\nChatGPT:\n\n{response['choices'][0]['message']['content']}")
