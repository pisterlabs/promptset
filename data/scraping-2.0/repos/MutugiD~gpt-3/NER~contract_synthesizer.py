import openai
from time import time,sleep
from uuid import uuid4
import streamlit as st 

openai.api_key = st.secrets['api']


prompt_input = 'NER/contract_prompt.txt'
prompts_output = 'Synthentic_data/NER/prompts'
completions_output = 'Synthentic_data/NER/completions'
logs = 'gpt3_logs/NER'

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)
        

dates = ['June 30, 2015', 
        '01-05-2015', 
        'April 3rd, 2011', 
        '02/05/2018'
        ]

prices = ['$4500000', 
         '£6200', 
         'JPY 290000', 
         '€26750'
         ]

uses = ['Corporate meetings', 
       'Co-working office', 
       'Virtual office', 
       'Training and development center'
       ]

areas = ['1,000 square meters', 
         '1200 Sq M', 
         '10,764 square feet', 
         '15700 Sq Ft'
         ]



def gpt3_completion(prompt, 
                    engine= "text-davinci-003",
                    temp=0.5, 
                    top_p=1.0, 
                    tokens=250, 
                    stop=None):  
                    
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                stop=stop)
            text = response['choices'][0]['text'].strip()
            #text = re.sub('\s+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            save_file(f'{logs}/%s' % filename, prompt + '\n\n==========\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)
            

if __name__ == '__main__':
    count = 0
    for date in dates:
        for price in prices:
            for use in uses :
                for area in areas:
                    count += 1
                    prompt = open_file(prompt_input)
                    prompt = prompt.replace('<date>', date)
                    prompt = prompt.replace('<price>', price)
                    prompt = prompt.replace('<use>',use)
                    prompt = prompt.replace('<area>', area)
                    prompt = prompt.replace('<<UUID>>', str(uuid4()))
                    print('\n\n', prompt)
                    completion = gpt3_completion(prompt)
                    outprompt = 'Date: %s\nPrice: %s\nUse: %s\nArea: %s\n\nOutput: ' % (date, price, use, area)
                    filename = (date + price + use + area).replace(' ','').replace('&','') + '%s.txt' % time()
                    save_file(f'{prompts_output}/%s' % filename, outprompt)
                    save_file(f'{completions_output}/%s' % filename, completion)
                    print('\n\n', outprompt)
                    print('\n\n', completion)
                    if count > 300:
                         exit()
    print(count)
            
            

