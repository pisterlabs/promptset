API_KEY = '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'
import openai
import os
os.environ['OPENAI_API_KEY'] = API_KEY
openai.api_key = os.environ['OPENAI_API_KEY']

keep_prompting = True
while keep_prompting:
    print('ReviewGenie : \t',end='')
    prompt = input('Hello, how can I help you today (type "exit" if done):\n\nUser: \t')
    print('\n')
    if prompt == 'exit':
        keep_prompting = False
    else:
        response = openai.Completion.create(engine = 'text-davinci-003', prompt=prompt,max_tokens = 200)
        print('ReviewGenie : \t',end='')
        print(response['choices'][0]['text'])
        print('\n')
