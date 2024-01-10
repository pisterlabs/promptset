import argparse
import os
import re
from openai import OpenAI
import time
import json

parser = argparse.ArgumentParser(description='Tests LLM\'s efficiency in generating code')

parser.add_argument('--data', type=str, required=True, help='human eval jsonl file path')
parser.add_argument('--output', type=str, required=True, help='output directory path')
parser.add_argument('--temp', type=float, default=0.8, help='temperature for LLM codegen')

args = parser.parse_args()

secret_key = "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"
client = OpenAI(api_key = secret_key)

motivation = "You are a python programmer, writing only the functions as described in the comments"

datafile = open(args.data)
data = [json.loads(l) for l in datafile.readlines()]
datafile.close()

for k, inst in enumerate(data):
    os.makedirs(args.output + f"/{k}/", exist_ok=True)

    for j in range(10):
        completion = client.chat.completions.create(
            model = "gpt-3.5-turbo",
            messages = [
                {"role": "system", "content": motivation},
                {"role": "user", "content": inst['prompt']} ],
            max_tokens = 1024,
            temperature = args.temp
        )
        time.sleep(20)

        out = completion.choices[0].message.content

        fw = open(args.output + f"/{k}/{j}.py", "w")
        fw.write(out)
        fw.close()
        
        print("Done {}, {}".format(k, j))

