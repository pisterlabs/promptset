import argparse
import os
import re
from openai import OpenAI
import time
import json
import subprocess

parser = argparse.ArgumentParser(description='Tests LLM\'s efficiency in generating postconditions')

parser.add_argument('--data', type=str, required=True, help='human eval jsonl file path')
parser.add_argument('--output', type=str, required=True, help='output directory path')
parser.add_argument('--temp', type=float, default=0.8, help='temperature for LLM codegen')
parser.add_argument('--runner', type=str, required=True, help='path to model runner')
parser.add_argument('--model', type=str, required=True, help='path to model')
parser.add_argument('--timeout', type=int, required=True, help='upper bound LLM codegen time')

args = parser.parse_args()

secret_key = "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"
client = OpenAI(api_key = secret_key)

datafile = open(args.data)
data = [json.loads(l) for l in datafile.readlines()]
datafile.close()

tmpfile = "/tmp/__tmpfile.py"

for k, inst in enumerate(data):
    os.makedirs(args.output + f"/{k}/", exist_ok=True)

    prompt = """You have the following code context, function stub and natural language specification (in the form of a code comment) for {}. When implemented, the function should comply with this natural language specification:
{}
Write a symbolic postcondition for {} consisting of exactly one assert statement. For variables, use only the function input parameters and a hypothetical return value, which we'll assume is stored in a variable return_val. If the postcondition calls any functions external to the program context, they should only be those from the functional subset of python. Although the postcondition should be less complex than the function itself, it should not be trivial. It should encapsulate an aspect of the function without implementing the function. The format of your response should be:
```CODE FOR EXACTLY ONE POSTCONDITION WITH ASSERT HERE```""".format(inst['entry_point'], inst['prompt'], inst['entry_point'])

    tmpfd = open(tmpfile, "w")
    tmpfd.write(prompt)
    tmpfd.close()

    for j in range(5):
        t = time.time()
        try:
            out = subprocess.run([args.runner, "-m", args.model, "-f", tmpfile,
                     "--temp", str(args.temp), "--prompt-cache", "/tmp/__cache.py", "--mlock"], 
                     capture_output=True, timeout=args.timeout)
        except subprocess.TimeoutExpired as e:
            out = e

        output = out.stdout.decode('utf-8')

        fw = open(args.output + f"/{k}/{j}.py", "w")
        fw.write(output)
        fw.close()
        
        print("Done {}, {} : {}".format(k, j, time.time()-t))

