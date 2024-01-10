import json
import re
from openai_req import OpenaiReq
import random
from tqdm import tqdm
import os
from multiprocessing import Pool
from termcolor import colored
random.seed(42)

MAX_SPLIT = 64
STEP = 4

def query(rank, prompts):
    print('Process rank {} PID {} begin...'.format(rank, os.getpid()))
    reqor = OpenaiReq()
    queries = prompts[int(len(prompts) * rank / MAX_SPLIT) : int(len(prompts) * (rank + 1) / MAX_SPLIT)]
    try:
        fout = open('outputs/rank_{}.json'.format(rank), 'w')
        if rank == 0:
            bar = tqdm(range(len(queries) // STEP + 1))
        else:
            bar = range(len(queries) // STEP + 1)
        for idx in bar:
            inputs = queries[idx * STEP : (idx + 1) * STEP]
            if len(inputs) == 0:
                break
            gpt_results = []
            for prompt in inputs:
                result, tag = reqor.req2openai(prompt, max_tokens = 512, stop = '\n\n')
                gpt_results.append(result[0])
            for prompt, res in zip(inputs, gpt_results):
                # print(res)
                fout.write(json.dumps({'prompt': prompt, 'response': res}) + '\n')
                fout.flush()
        fout.close()
    except Exception as err:
        print(Exception, err)

if __name__=='__main__':
    prompts = json.load(open('prompts.json'))
    os.makedirs("outputs", exist_ok=False)
    print("number of prompts: {}".format(len(prompts)))
    print('Parent process %s.' % os.getpid())
    p = Pool(MAX_SPLIT)
    for i in range(MAX_SPLIT):
        p.apply_async(query, args=(i, prompts))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')