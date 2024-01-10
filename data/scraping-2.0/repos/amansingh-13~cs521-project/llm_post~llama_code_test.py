import argparse
import os
import re
from openai import OpenAI
import time
import json
import subprocess

parser = argparse.ArgumentParser(description='Tests LLM\'s efficiency in generating code')

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

    prompt = inst['prompt']

    tmpfd = open(tmpfile, "w")
    tmpfd.write(prompt)
    tmpfd.close()

    for j in range(10):
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

