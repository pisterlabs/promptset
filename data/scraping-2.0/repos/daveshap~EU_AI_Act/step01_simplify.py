import os
import textwrap
from halo import Halo
import openai
from time import time, sleep



###     file operations



def save_yaml(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as file:
        yaml.dump(data, file, allow_unicode=True)



def open_yaml(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data



def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)



def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
        return infile.read()




def chunk_string(s):
    lines = s.split('\n')
    return ['\n'.join(lines[i:i+100]) for i in range(0, len(lines), 100)]



def save_strings(strings, folder):
    for i, s in enumerate(strings, 1):
        filepath = os.path.join(folder, f'file{i:05}.txt')
        save_file(filepath, s)


###     chatbot functions



def chatbot(messages, model="gpt-4-0613", temperature=0):
#def chatbot(messages, model="gpt-3.5-turbo-0613", temperature=0):
    openai.api_key = open_file('key_openai.txt')
    max_retry = 7
    retry = 0
    while True:
        try:
            response = openai.ChatCompletion.create(model=model, messages=messages, temperature=temperature)
            text = response['choices'][0]['message']['content']
            return text, response['usage']['total_tokens']
        except Exception as oops:
            print(f'\n\nError communicating with OpenAI: "{oops}"')
            if 'maximum context length' in str(oops):
                a = messages.pop(1)
                print('\n\n DEBUG: Trimming oldest message')
                continue
            retry += 1
            if retry >= max_retry:
                print(f"\n\nExiting due to excessive errors in API: {oops}")
                exit(1)
            print(f'\n\nRetrying in {2 ** (retry - 1) * 5} seconds...')
            sleep(2 ** (retry - 1) * 5)



if __name__ == '__main__':
    act = open_file('act.txt')
    system = open_file('system_summarize.txt')
    chunks = chunk_string(act)
    summaries = list()
    for chunk in chunks:
        messages = [{'role': 'user', 'content': chunk}, {'role': 'system', 'content': system}]
        
        spinner = Halo(text='Summarizing next chunk...', spinner='dots')
        spinner.start()
        
        response, tokens = chatbot(messages)
        
        spinner.stop()
        
        formatted_lines = [textwrap.fill(line, width=120, initial_indent='    ', subsequent_indent='    ') for line in response.split('\n')]
        formatted_text = '\n'.join(formatted_lines)
        print('\n\n\n\n', formatted_text)
        summaries.append(response)
    save_strings(summaries, 'summaries')