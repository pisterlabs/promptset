import argparse
import os
import re
from openai import OpenAI
import time

parser = argparse.ArgumentParser(description='Tests LLM\'s efficiency in generating asserts')

parser.add_argument('--abench', type=str, required=True, help='assert benchmark directory')
parser.add_argument('--temp', type=float, default=0.8, help='temperature for LLM codegen')

args = parser.parse_args()

secret_key = "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"
client = OpenAI(api_key = secret_key)

motivation = "You are a C programmer, skilled in writing useful and correct assertions"

def parse(lines):
    source, target = [], []
    asrt_lnos, cmnt_lnos = [], []

    # improve this
    for i,l in enumerate(lines):
        if re.search("assert[(].+[)]", l):
            asrt_lnos.append(i)
        elif re.search("//", l):
            cmnt_lnos.append(i)
    return asrt_lnos, cmnt_lnos


for f in os.listdir(args.abench):
    fd = open(os.path.join(args.abench, f), "r")
    
    lines = fd.readlines()
    asrt_lnos, cmnt_lnos = parse(lines)
    assert len(asrt_lnos) == len(cmnt_lnos), "Dataset formatting error : {}".format(f)

    for i, al in enumerate(asrt_lnos):
        cl = cmnt_lnos[i]
        assert al > cl

        prompt = "".join(lines[:cl+1] + ["\n"])
        truth = "".join(lines[cl+1:al+1])
        
        completion = client.chat.completions.create(
            model = "gpt-3.5-turbo",
            messages = [
                {"role": "system", "content": motivation},
                {"role": "user", "content": prompt} ],
            max_tokens = 1024,
            temperature = args.temp
        )
        time.sleep(20)

        out = prompt + completion.choices[0].message.content
        olines = out.split('\n')
        oasrt_lnos, ocmnt_lnos = parse(olines)
        assert ocmnt_lnos[i] == cl
        if(len(oasrt_lnos) > i):
            output = "\n".join(olines[ocmnt_lnos[i]+1:oasrt_lnos[i]+1])
        else:
            output = "\n".join(olines[ocmnt_lnos[i]+1:])

        print("--FILENAME--")
        print(f)
        print("--TRUTH--")
        print(truth)
        print("--OUTPUT--")
        print(output)

    fd.close()
