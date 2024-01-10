import argparse
import os
import re
from openai import OpenAI
import time
import json

parser = argparse.ArgumentParser(description='Tests LLM\'s efficiency in generating postconditions')

parser.add_argument('--data', type=str, required=True, help='human eval jsonl file path')
parser.add_argument('--output', type=str, required=True, help='output directory path')
parser.add_argument('--temp', type=float, default=0.8, help='temperature for LLM codegen')

args = parser.parse_args()

secret_key = "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"
client = OpenAI(api_key = secret_key)


datafile = open(args.data)
data = [json.loads(l) for l in datafile.readlines()]
datafile.close()

for k, inst in enumerate(data):
    if(k <= 29):
        continue

    os.makedirs(args.output + f"/{k}/", exist_ok=True)

    prompt = """You have the following code context, function stub and natural language specification (in the form of a code comment) for {}. When implemented, the function should comply with this natural language specification:
{}
Write a symbolic postcondition for {} consisting of exactly one assert statement. For variables, use only the function input parameters and a hypothetical return value, which we'll assume is stored in a variable return_val. If the postcondition calls any functions external to the program context, they should only be those from the functional subset of python. Although the postcondition should be less complex than the function itself, it should not be trivial. It should encapsulate an aspect of the function without implementing the function. The format of your response should be:
```CODE FOR EXACTLY ONE POSTCONDITION WITH ASSERT HERE```""".format(inst['entry_point'], inst['prompt'], inst['entry_point'])

    for j in range(5):
        completion = client.chat.completions.create(
            model = "gpt-3.5-turbo",
            messages = [
                {"role": "user", "content": prompt} ],
            max_tokens = 1024,
            temperature = args.temp
        )
        time.sleep(20)

        out = completion.choices[0].message.content

        fw = open(args.output + f"/{k}/{j}.py", "w")
        fw.write(out)
        fw.close()

        print("Done {}, {}".format(k, j))

