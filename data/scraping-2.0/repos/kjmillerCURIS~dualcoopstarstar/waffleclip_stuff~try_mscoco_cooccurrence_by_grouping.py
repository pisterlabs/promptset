import os
import sys
import numpy as np
import pickle
import random
from tqdm import tqdm
from openai import OpenAI
from openai_utils import OPENAI_API_KEY


OUT_FILENAME = 'mscoco_grouping.pkl'
LIKELIHOOD_STRS = ['very likely', 'likely', 'unlikely', 'very unlikely']
VERBOSE = True


def get_mscoco_classnames():
    return ['horse', 'donkey', 'pig', 'apple', 'banana', 'hay', 'saddle', 'computer', 'mouse', 'keyboard', 'boots', 'dress', 'tshirt']
#    assert(False) #KEVIN


def build_prompts(classnames, random_seed=0):
    random.seed(random_seed)
    prompts = []
    for classname in classnames:
        prompt = 'Please group the following objects by the likelihood that they co-occur with "%s".'%(classname)
        prompt = prompt + ' The groupings should be called %s.'%(','.join(['"%s"'%(z) for z in LIKELIHOOD_STRS]))
        prompt = prompt + ' The output should be formatted to look like this:\n'
        example_lines = []
        for k, likelihood_str in enumerate(LIKELIHOOD_STRS):
            example_lines.append('"%s":"obj%d","obj%d",...'%(likelihood_str, 2*k, 2*k+1))

        prompt = prompt + '\n'.join(example_lines) + '\n'
        prompt = prompt + 'where things like obj1, obj2, etc. are actual object names that are given. Here are the object names to group:'
        prompt = prompt + ','.join(['"%s"'%(y) for y in random.sample(classnames, len(classnames)) if y != classname])
        prompts.append(prompt)

    return prompts


#should return d[<classA>][<classB>] = <likelihood_str>
#FIXME: I'm making this as straightforward and unforgiving as possible. Might need to make it more robust.
def postprocess(completion, classnames):
    outputs = [c.text for c in completion.choices]
    assert(len(outputs) == len(classnames))
    out = {}
    for output, classname in zip(outputs, classnames):
        out[classname] = {}
        if VERBOSE:
            print('"%s" output:'%(classname))
            print(output)

        lines = output.split('\n')
        assert(len(lines) == len(LIKELIHOOD_STRS))
        already_seen = []
        for line in lines:
            ss = line.split('":')
            assert(len(ss) == 2)
            assert(ss[0][0] == '"')
            assert(ss[0][1:] in LIKELIHOOD_STRS)
            assert(ss[0][1:] not in already_seen)
            likelihood_str = ss[0][1:]
            already_seen.append(likelihood_str)
            if len(ss[1]) == 0:
                continue

            assert(ss[1][0] == '"')
            assert(ss[1][-1] == '"')
            grouped_classnames = ss[1][1:-1].split('","')
            for grouped_classname in grouped_classnames:
                assert(grouped_classname in classnames)
                assert(grouped_classname != classname)
                assert(grouped_classname not in out[classname])
                out[classname][grouped_classname] = likelihood_str

        assert(all([other_classname in out[classname] for other_classname in classnames if other_classname != classname]))

    return out


def get_client():
    client = OpenAI(api_key=OPENAI_API_KEY)
    return client


def query_LLM(prompts, client):
    completion = client.completions.create(model='text-davinci-003', prompt=prompts, temperature=0., max_tokens=300)
    return completion


def try_mscoco_cooccurrence_by_grouping():
    client = get_client()
    classnames = get_mscoco_classnames()
    prompts = build_prompts(classnames, random_seed=3)
    completion = query_LLM(prompts, client)
    out = postprocess(completion, classnames)
    import pdb
    pdb.set_trace()
    with open(OUT_FILENAME, 'wb') as f:
        pickle.dump(out, f)


if __name__ == '__main__':
    try_mscoco_cooccurrence_by_grouping()
