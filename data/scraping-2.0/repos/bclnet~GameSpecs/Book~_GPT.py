import os, time
from openai import OpenAI

# open ai client
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# yeild paths recursivly and in order
def yieldPaths(folder, extension):
    files = {os.path.basename(f.path):f for f in os.scandir(folder) if f.is_dir() or f.path.endswith(extension)}
    _intro = files.pop('_intro' + extension, None)
    if _intro: yield _intro.path
    for f in files.values():
        if f.is_dir():
            for x in yieldPaths(f.path, extension): yield x
        else: yield f.path

# openai chat
messages = [{'role': 'system', 'content': 'You are a intelligent assistant.'}]
def chat(content):
    print(f'Python: {content}') 
    messages.append({'role': 'user', 'content': content}) 
    completion = client.chat.completions.create(model='gpt-3.5-turbo-1106', messages=messages)
    reply = completion.choices[0].message.content
    print(f'ChatGPT: {reply}') 
    messages.append({'role': 'assistant', 'content': reply}) 

# process .gpt files
for path in yieldPaths('book', '.gpt'):
    with open(path, 'r') as f: reply = chat(f.read())
    # sleep for a second
    time.sleep(1)