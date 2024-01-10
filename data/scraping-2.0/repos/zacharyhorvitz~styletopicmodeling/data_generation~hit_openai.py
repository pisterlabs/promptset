
import json
import os
import random
from tqdm import tqdm
import time

from openai import OpenAI

client = OpenAI()


def get_author_data(*, author_name, author_directory, shard='train'):
    clean_name = author_name.replace(":","").replace(" ", "_")
    with open(os.path.join(author_directory, clean_name, f'{shard}.txt'), 'r') as f:
        lines = f.readlines()
        input_data = [line.strip().split('\t') for line in lines]
    return input_data

def hit_openai(samples):

    message = 'Here are some example emails written by the same author: \n'
    for sample in samples:
        message += '{"email body": "' + sample + '"}\n'
    message += "Can you write another short email that could also have been written by the same author, in the same style? Please only include the body of the email, not the subject line or any other metadata."
    print(message)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": message}
        ]
    )
    return response.choices[0].message.content

if __name__ == '__main__':

    random.seed(42)

    DATA_SRC = '/burg/nlp/users/zfh2000/enron_processed.json'

    with open(DATA_SRC, 'r') as in_file:
        data = json.load(in_file)

    with open('authors_splits.json', 'r') as in_file:
        splits = json.load(in_file)

    train = splits['train']

    sampled_authors = random.sample(train, 100)

    
    authors_to_emails = {a:[] for a in sampled_authors}

    random.shuffle(data)

    for d in data:
        if d['info']['from'] in authors_to_emails:
            authors_to_emails[d['info']['from']].append(d['text'])


    chat_gpt_out = []

    with open('chat_gpt_out.jsonl', 'w+') as out_file:
        for author in tqdm(authors_to_emails):
            for i in range(5):
                samples = random.sample(authors_to_emails[author], min(3, len(authors_to_emails[author])))

            
                while True:
                    try: 
                        result = hit_openai(samples)
                        break
                    except: 
                        print("Service unavailable, retrying...")
                        time.sleep(20)
                        continue
                
                print(result)
                chat_gpt_out.append(dict(author=author, samples=samples, result=result))
                out_file.write(json.dumps(dict(author=author, samples=samples, result=result)) + '\n')

    import pdb; pdb.set_trace()