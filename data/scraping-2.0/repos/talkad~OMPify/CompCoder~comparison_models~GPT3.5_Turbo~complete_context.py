from openai import OpenAI
from tqdm import tqdm
import json
from transformers import GPT2Tokenizer

def remove_pragma(code):
    buf = []

    for line in code.split('\n'):
        if line.lstrip().startswith('#pragma'):
            continue

        buf.append(line)

    return '\n'.join(buf)


start_idx = 0
context = 600

with open("openai.key", "r") as f:
    key = f.read()

client = OpenAI(api_key=key)

prompt = """Complete the following code:
{}
"""

test_file = '/home/talkad/LIGHTBITS_SHARE/HPCorpus_omp.jsonl'

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

with open(test_file, 'r') as f, open(f'context_{context}.jsonl', 'w') as out:
    samples = f.readlines()[:500]

    for idx, line in tqdm(enumerate(samples[start_idx:])):
        sample = json.loads(line)
        
        code = remove_pragma(sample["code"])
        tokens = tokenizer.encode(code, max_length=context, truncation=True)
        context_window = tokenizer.decode(tokens)

        try:
            response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt.format(context_window)}
            ]
            )

            output = {'label': code,
                    'pred': context_window + response.choices[0].message.content}

            out.write(json.dumps(output) + '\n')
        except Exception as e:
            print(f'failed at sample {start_idx+idx}')

