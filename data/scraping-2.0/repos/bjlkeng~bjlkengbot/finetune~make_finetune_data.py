""" Generate fine-tuning data for GPT by generating a question to the chunked
blog post snippets. 
"""
import time
import jsonlines
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

filename = '../crawler/briankeng-2023-06-09.jsonl'
params = {
    'temperature': 0.2,
    'model_name': 'gpt-3.5-turbo',
    'max_tokens': 2000,
}

PROMPT = \
"""Write a concise question in as few words as possible to the author in the second person that has the following TEXT as the answer.

### TEXT ###
"""
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)

# Call OpenAI API
llm = ChatOpenAI(**params)

# Read in json files
data = []
with jsonlines.open(filename) as reader:
    for obj in reader:
        title = obj['title']
        content = obj['content']
        url = obj['url']
        chunks = text_splitter.split_text(content)
        for chunk in chunks:
            if len(chunk) < 20:
                continue
            data.append({'content': chunk, 'url': url, 'title': title})

output = []
for i, d in enumerate(data):
    content = d['content']
    url = d['url']
    title = d['title']

    # Pick the largest paragraph that doesn't start with "Related Posts"
    splits = sorted(content.split('\n\n'), key=len, reverse=True)
    splits = [s for s in splits if not s.strip().startswith('Related Posts')]
    if not splits:
        continue
    text = splits[0]

    prompt = PROMPT + text

    # Add required pre-processing for OpenAI fine-tuning
    question = "QUESTION: " + llm.predict(prompt) + "\n\n###\n\n"
    text = " " + text + " END"
    print('----------------------------------------')
    print(f'{question}')
    print(f'{text}')
    output.append({'prompt': question, 'completion': text})

    # Sleep every 100 requests
    if (i + 1) % 40 == 0:
        print('Sleeping...')
        time.sleep(60)

# jsonlines write out file
with jsonlines.open('briankeng-2023-06-09-finetune.jsonl', mode='w') as writer:
    writer.write_all(output)

# Write out text file
with open('briankeng-2023-06-09-finetune.txt', 'w') as f:
    for o in output:
        f.write(f'QUESTION: {o["prompt"]}\n')
        f.write(f'ANSWER: {o["completion"]}\n')
        f.write('\n\n')
