"""augment.py 

Data augmentation routines.

Author: Gyu-min Lee 
his.nigel at gmail dot com 
"""

import os
import sys
import time
import json 
import random

from collections import deque
from datetime import datetime

import yaml
import openai

from openai.error import OpenAIError 
from tqdm import tqdm 

random.seed(612)

def _get_src(jsonPath):
    with open(jsonPath) as f:
        data = json.load(f)
    return data

def _make_qano_list(listPath) -> list:
    with open(listPath) as f:
        result = f.read().split('\n')
    return [int(i.strip()) for i in result if not i.strip().startswith('#')]

def _get_elements(jsonPath):
    data = _get_src(jsonPath)
    data = data['data'].values()
    return [j for i in data for j in i]

def _make_data_dict(data):
    return dict((e['qano'], e) for e in data)
    
def _make_data_list(data, doMergeSplits: bool=True) -> dict:
    """
    Construct a list for dataset splits. 
    Params:
        data: json data parsed with _get_src()
        doMergeSplits (bool): if True, all splits will be merged into one,
                                  losing its source split and assigned to 
                                  a single split.
                                  To maintain the splits, set as False.
    Returns:
        dict of list, where key is the original split and the value is the 
                      list of split data. If doMergeSplits is True, all data 
                      are returned as members of the train split.
    """ 
    
    train, dev, test = list(data['data'].values())
    if doMergeSplits:
        train.extend(dev)
        dev = list()
        train.extend(test)
        test = list()
    return {
            "train": train,
            "dev": dev, 
            "test": test
            }

def _load_prompt(promptPath):
    with open(promptPath) as f:
        data = f.read()
    return data

def _call(data: dict, 
          promptTemplate: str,
          gptArgs:dict,
          prebuilt: dict=dict(),
          ) -> dict:
    if data['qano'] in prebuilt.keys():
        return prebuilt[data['qano']]
    question = '\n'.join(data['question'])
    data_intext = '\n'.join([
            f"기관: {data['organization']}",
            f"제목: {data['title']}",
            f"내용: \n{question}",
            ])
    prompt = promptTemplate.replace('${var1}', data_intext)

    if gptArgs['model'] in "gpt-4, gpt-4-0314, gpt-4-32k, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-0301".split(', '):
        response = openai.ChatCompletion.create(
                model=gptArgs['model'],
                messages=[{
                    "role": "user",
                    "content": prompt
                    }],
                temperature=gptArgs['temperature'],
                max_tokens=gptArgs['max_tokens'],
                top_p=gptArgs['top_p'],
                frequency_penalty=gptArgs['frequency_penalty'],
                presence_penalty=gptArgs['presence_penalty'],
                )
    else:
        response = openai.Completion.create(
                model=gptArgs['model'],
                prompt=prompt,
                temperature=gptArgs['temperature'],
                max_tokens=gptArgs['max_tokens'],
                top_p=gptArgs['top_p'],
                frequency_penalty=gptArgs['frequency_penalty'],
                presence_penalty=gptArgs['presence_penalty'],
                )
    
    data['response'] = response
    return data 

def _save(obj, outname: str='out.json'):
    with open(outname, 'w') as f:
        json.dump(obj, f, ensure_ascii=False,
                  indent=4)

def augment(args, creds, doTest: bool=False):
    didIterateWithNoError = True
    promptTemplate = _load_prompt(args['promptPath'])
    gptArgs = args['gptArgs']
    
    openai.api_key = creds['OpenAI']
    
    builtData = dict()
    
    if args['continueFromTemp'] and '.augment_temp.json' in os.listdir(args['outputPath']):
        builtData = _get_elements(os.path.join(args['outputPath'], '.augment_temp.json'))
        builtData = _make_data_dict(builtData)
    
    data = _get_src(args['inPath'])
    trainlist = _make_data_list(data)['train']
    print(f"Loaded: {len(trainlist)}")
    
    if 'allowListPath' in args.keys():
        allowlist = _make_qano_list(args['allowListPath'])
        if len(allowlist) > 0:
            print(f"Allowing: {len(allowlist)}")
            trainlist = [i for i in trainlist if i['qano'] in allowlist]
    if 'ignoreListPath' in args.keys():
        ignorelist = _make_qano_list(args['allowListPath'])
        if len(ignorelist) > 0:
            trainlist = [i for i in trainlist if i['qano'] not in ignorelist]
    print(f"Reduced with allowing and ignoring: {len(trainlist)}")

    if doTest:
        trainlist = random.sample(trainlist, min(3, len(trainlist)))
        print(f"Will run only with the sample of {len(trainlist)} for testing.")
        gptArgs['max_tokens'] = 100
        print("Also set 'max_tokens' as 100 for testing.")
        
    trainlist = deque(trainlist)

    trainlistWithResponse = list()
    with tqdm(total=len(trainlist), desc="getting responses...") as bar:
        d = trainlist.popleft()
        while True:
            try:
                trainlistWithResponse.append(_call(d, promptTemplate, 
                                                   gptArgs=args['gptArgs'], 
                                                   prebuilt=builtData,
                                                   ))
                bar.update(1)
                if len(trainlist) < 1: break
                d = trainlist.popleft()
            except OpenAIError as e:
                print(f"Encountered OpenAI-side error:\n\t{e}")
                if 'Please reduce the length of the messages or completion.' in str(e):
                    # a heuristic check for token length-related issue 
                    # as for OpenAI this issue does not get a special error ID :/
                    print("Skipping the generation for the data.")
                    print("Check the one with `response: null`")
                    d['response'] = None
                    trainlistWithResponse.append(d)
                    bar.update(1)
                    if len(trainlist) < 1: break
                    d = trainlist.popleft()
                else:    
                    print("Will try again after a minute...")
                    time.sleep(60)
                continue
            except KeyboardInterrupt:
                print("Stopping upon keyboard interrupt.")
                print(f"Want to save {len(trainlistWithResponse)} responses got?")
                print(f"(entries with no response won't be saved)")
                if input("Yes/No: ").lower() == 'n':
                   print(f"will discard fetched data") 
                   trainlistWithResponse = list()
                break
            except BaseException as e:
                print(f"The following error occurred while iterating:\n\t{e}")
                print("Will save the items as `out_temp.json` with responses; consider continuing from there.")
                didIterateWithNoError = False
                break
        
    if trainlistWithResponse != list():
        data['data']['train'] = trainlistWithResponse
        data['dataset'] = '민원_openapi_200101-230531_aug'
        if doTest: data['dataset'] += 'test'
        data['lastupdate'] = datetime.now().isoformat()
        data['datacount'] = {
                "train": len(data['data']['train']),
                "dev": len(data['data']['dev']), 
                "test": len(data['data']['test']), 
                "total": (
                         len(data['data']['train'])
                          + len(data['data']['dev'])
                          + len(data['data']['test'])
                        )
                }
        outName = 'augmented_test' if doTest else args['outputName'] 
        _save(data, os.path.join(args['outputPath'], outName+'.json' if didIterateWithNoError else '.augment_temp.json'))