import os
import openai

def chat(question):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a chatbot"},
            {"role": "user", "content": f"{question}"}
        ]
    )  
    return response['choices'][0]['message']['content']

openai.api_key = os.getenv("OPENAI_API_KEY")

opList = ['quit', '?', 'save', 'shell', 'sh']

user = input('user:')
response = None
question = None
while True:
    command = input('command:').lower()
    args = command.split()
    op = args[0]
    tail = ' '.join(args[1:])
    if op not in opList:
        print(f'Operation error, only {opList} allowed!')
        continue
    if op == 'quit': break
    if op == 'sh' or op == 'shell':
        os.system(tail)
    if op == '?':
        question = tail
        response = chat(question)
        print(response)
    if op == 'save':
        if response is None:
            print('errro: no response to save!')
        else:
            fname = args[1]
            with open(fname, 'a+', encoding='utf-8') as file:
                file.write(f"\n\n## {user}: {question}\n\nChatGPT:\n\n{response}")
 