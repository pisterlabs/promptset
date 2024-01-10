import json
import os
import openai
import time
from datetime import datetime
import argparse
from tqdm import tqdm
from prompts import math_prompt
from prompts.plancode_util_v2 import *
import jsonlines as jsl
from collections import OrderedDict, Counter
from tool import *
from prompts.prep_reflexion.actor_prompt_utils import PromptStr
import random 
from collections import OrderedDict
# from tenacity import retry, wait_chain, wait_fixed
# from itertools import combinations
from pathlib import Path


# after 6 times of retry, it raises exception. waits between retries are specified inside the `wait_chain`
# @retry(wait=wait_chain(*[wait_fixed(3) for i in range(5)])) #defining backoff for retrying.
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def get_user_assistant_messages(system_message: str, 
                                user_message: str, 
                                assistant_message: str):
    '''
    This function is used to convert the prompt into the message format used by OpenAI Chat API.
    '''
    messages = []
    messages.append({"role": "system", "content": system_message})
    split_user_messages = user_message.split('\n'*4)
    split_assistant_messages = assistant_message.split('\n'*4) # delim==4*\n... 
    for i in range(len(split_user_messages)): # user messages and assistant messages are paired... actually. This should have been `zip()`.
        question = split_user_messages[i]
        answer = split_assistant_messages[i]
        messages += [
            {"role": "user", "content": f"{question}"},
            {"role": "assistant", "content": f"{answer}"},
        ]
    return messages


def get_cot_prompt(data: dict, backbone: str, hint:str=''):
    '''
    This function is used to generate the CoT prompt.
    '''
    if backbone == 'gpt4' or backbone == 'gpt4turbo' or 'gpt4turbo':
        system_message = math_prompt.GPT4_COT_SYSTEM
        user_message = math_prompt.GPT4_COT_USER
        assistant_message = math_prompt.GPT4_COT_ASSISTANT
    elif backbone == 'chatgpt':
        system_message = math_prompt.TURBO_COT_SYSTEM
        user_message = math_prompt.TURBO_COT_USER
        assistant_message = math_prompt.TURBO_COT_ASSISTANT

    messages = get_user_assistant_messages(
        system_message, user_message, assistant_message)
    question_message = data['question']
    if hint:
        question_message += f" ({hint})"
    # print(question_message)
    messages += [{"role": "user", "content": f"Question: {question_message}"}]


    return messages



def get_pal_prompt(data: dict, backbone: str, hint:str=''):
    '''
    This function is used to generate the PAL prompt.
    '''
    if backbone == 'gpt4' or backbone == 'gpt4turbo':
        system_message = math_prompt.GPT4_PAL_SYSTEM
        user_message = math_prompt.GPT4_PAL_USER
        assistant_message = math_prompt.GPT4_PAL_ASSISTANT

        messages = get_user_assistant_messages(
            system_message, user_message, assistant_message)

        question_message = data['question']
        if hint:
            question_message += f" ({hint})"
        messages += [{"role": "user",
                      "content": f"Question: {question_message}\n\n# solution in Python"}]

    elif backbone == 'chatgpt':
        system_message = math_prompt.TURBO_PAL_SYSTEM
        user_message = math_prompt.TURBO_PAL_USER
        assistant_message = math_prompt.TURBO_PAL_ASSISTANT

        messages = get_user_assistant_messages(
            system_message, user_message, assistant_message)

        question_message = data['question']
        if hint:
            question_message += f" ({hint})"
        messages += [{"role": "user",
                      "content": f"Answer the following question in Python: {question_message}"}]
    # print(question_message)
    return messages


def get_select_prompt(data: dict, cot_solution: list, pal_solution: list, backbone: str):
    '''
    This function is used to generate the selection prompt.
    '''
    if backbone == 'gpt4' or backbone == 'gpt4turbo':
        system_message = math_prompt.GPT4_SELECT_SYSTEM
        user_message = math_prompt.GPT4_SELECT_USER
        assistant_message = math_prompt.GPT4_SELECT_ASSISTANT
    elif backbone == 'chatgpt':
        system_message = math_prompt.TURBO_SELECT_SYSTEM
        user_message = math_prompt.TURBO_SELECT_USER
        assistant_message = math_prompt.TURBO_SELECT_ASSISTANT
    messages = get_user_assistant_messages(
        system_message, user_message, assistant_message)

    try:
        pal_solution_lines_strip = [l.strip for l in pal_solution.split('\n')]
        docstring_idxs = [i for i, x in enumerate(pal_solution_lines_strip) if x == '"""' or x == "'''"]
        dsstart, dsend = min(docstring_idxs), max(docstring_idxs)

        pallines = [l for l in pal_solution.split('\n')]
        pal_generated = "\n".join(pallines[:dsstart] + pallines[dsend+1:])
    except Exception as e:
        pal_generated = pal_solution[0]

    if cot_solution[0].startswith('Answer:'):
        cot_generated = cot_solution[0]
    else:
        cot_generated = 'Answer:\n' + cot_solution[0]

    user_message = f'''Math problem: {data['question'].strip()}

(A)
{cot_generated.strip()}

(B)
{pal_generated.strip()}

Which of the above two choices can correctly answer the math problem?'''

    messages += [{"role": "user", "content": user_message}]

    return messages


def query_cot(data: dict, key: str, cot_temperature: float, backbone: str, hint:str='', n=1, seed=777):
    '''
    This function is used to query OpenAI for CoT solutions.

    Args:
        data: a dict containing the question and answer
        key: the OpenAI API key
        cot_temperature: the temperature used in CoT
        backbone: ChatGPT or GPT-4

    Returns:
        completions: a list containing the CoT solution
    '''
    query_message = get_cot_prompt(data, backbone=backbone, hint=hint)
    if backbone == 'gpt4':
        model_name = 'gpt-4'
    elif backbone == 'gpt4turbo':
        model_name  = 'gpt-4-1106-preview'
    elif backbone == 'chatgpt':
        model_name = 'gpt-3.5-turbo'

    completions = []
    cot_solution = completion_with_backoff(
            api_key=key,
            model=model_name,
            max_tokens=500,
            stop='\n\n\n',
            messages=query_message,
            temperature=cot_temperature,
            top_p=1.0,
            seed=seed,
            n=n)
    if n ==1 :
        completions = [cot_solution['choices'][0]['message']['content']]
    else:
        completions = [cot_solution['choices'][i]['message']['content'] for i in range(n)]
    return completions, query_message


def _query(key, model_name='gpt-3.5-turbo', max_tokens=2048, stop='</end>', messages=None, temperature=0., top_p=1.0, n=1, mode='plan', k_fewshot:int=0, seed=777): # mode = plan or code
    resp = openai.ChatCompletion.create(api_key=key,
                                model=model_name,
                                max_tokens=max_tokens,
                                stop=stop,
                                messages=messages,
                                temperature=temperature,
                                top_p=top_p,
                                n=n, seed=seed)
    if n ==1:
        content = resp['choices'][0]['message']['content'] # str
        if mode == 'plan':
            plan = postprocess_plan(content) # it will complain when failing
            return plan
        elif mode == 'code':
            code = postprocess_code(content)
            return code 
    else: # n>1
        contents = [ch['message']['content'] for ch in resp['choices']]
        postprocess = postprocess_plan if mode=='plan' else postprocess_code
        res_strs = [postprocess(c) for c in contents]
        return res_strs

def query_plancode(data: dict, key: str='', plan_temperature: float=.0, code_temperature: float=.0, backbone: str='gpt-3.5-turbo', k_fewshot:int=0, hint:str='', n=1, seed:int=777):
    '''
    PAL variant: 1. generate planning for the given question 2. based on 1, generate code like PAL does.

    args:
        mostly same arguments with `query_pal()` below
    returns: 
        completions: Sequence[
                code_solution:str
                ]
    '''
    # specify model
    if backbone == 'gpt4':
        model_name = 'gpt-4'
    elif backbone == 'gpt4turbo':
        model_name = 'gpt-4-1106-preview'
    elif backbone == 'chatgpt':
        model_name = 'gpt-3.5-turbo'
        print(f'gpt-3.5 uses k_fewshot=8 as default (p2c fs-prompting)')
    if model_name.startswith('gpt-4'): # k_fewshot==8 is default.
        print(f'gpt-4 uses k_fewshot=5 as default (p2c fs_prompting)')
        k_fewshot = 5
    elif model_name.startswith('gpt-3.5-turbo'):
        k_fewshot = 8 

    # generate plan (retry included)
    plan_query_msg = get_plan_prompt(data, k_fewshot=k_fewshot, hint=hint)
    plan = _query(key, model_name=model_name, max_tokens=1024, stop='Question: ', messages=plan_query_msg, temperature=plan_temperature, top_p=1.0, n=1, mode='plan', seed=seed)

    if plan:
        code_query_msg = get_plan2code_prompt(data, plan=plan, k_fewshot=k_fewshot, hint=hint)
        code = _query(key, model_name=model_name, max_tokens=1024, stop='Question: ', messages=code_query_msg, temperature=code_temperature, top_p=1.0, n=n, mode='code', seed=seed)#, 
        if not code:
            return [None], [plan], {'codequery': code_query_msg, 'planquery': plan_query_msg}
        else: 
            return [code] if n==1 else code, [plan], {'codequery': code_query_msg, 'planquery': plan_query_msg}
    else:
        return None, None, {'codequery': code_query_msg, 'planquery': plan_query_msg}


def query_pal(data: dict, key: str, pal_temperature: float, backbone: str, hint:str='', n=1, seed=777):
    '''
    This function is used to query OpenAI for PAL solutions.

    Args:
        data: a dict containing the question and answer
        key: the OpenAI API key
        pal_temperature: the temperature used in PAL
        backbone: ChatGPT or GPT-4

    Returns:
        completions: a list containing the PAL solution
    '''
    query_message = get_pal_prompt(data, backbone=backbone, hint=hint)
    if backbone == 'gpt4':
        model_name = 'gpt-4'
    elif backbone == 'gpt4turbo':
        model_name = 'gpt-4-1106-preview'
    elif backbone == 'chatgpt':
        model_name = 'gpt-3.5-turbo'
    completions = []
    pal_solution = completion_with_backoff(
                                api_key=key,
                                model=model_name,
                                max_tokens=500,
                                stop='\n\n\n',
                                messages=query_message,
                                temperature=pal_temperature,
                                top_p=1.0,
                                seed=777,
                                n=n)

    if n ==1:
        completions.extend([choice['message']['content']
                        for choice in pal_solution['choices']]) # wtf this code...
        completions = completions[:1]
    else: # this line might not be compatible with self-consistency setting in the original code
        completions = [pal_solution['choices'][i]['message']['content'] for i in range(n)]
    return completions, query_message

def query_selection(data: dict, key: str, cot_solution: list, pal_solution: list, backbone: str):
    '''
    This function is used to query OpenAI for selection solutions.

    Args:
        data: a dict containing the question and answer
        key: the OpenAI API key
        cot_solution: a list containing the CoT solution
        pal_solution: a list containing the PAL solution
        backbone: ChatGPT or GPT-4

    Returns:
        completions: a list containing the selection solution
    '''
    selection_message = get_select_prompt(
        data, cot_solution, pal_solution, backbone=backbone)
    if backbone == 'gpt4':
        model_name = 'gpt-4'
    elif backbone == 'gpt4turbo':
        model_name = 'gpt-4-1106-preview'
    elif backbone == 'chatgpt':
        model_name = 'gpt-3.5-turbo'
    completions = []
    selection_solution = completion_with_backoff(
        api_key=key,
        model=model_name,
        max_tokens=200,
        stop='\n\n',
        messages=selection_message,
        temperature=0.,
        top_p=1.0,
        n=1)
    
    completions.extend([choice['message']['content']
                            for choice in selection_solution['choices']])
    completions = completions[:1]
    return completions

def query_actor_selection(data: dict, 
                          prompt_f: str, 
                          key: str, 
                          backbone: str)->tuple: # str(hint:plain english), str(method acronym), list(msgs)
    if backbone == 'chatgpt':
        model_name = 'gpt-3.5-turbo'
    elif backbone == 'gpt4':
        model_name = 'gpt-4'
    elif backbone == 'gpt4turbo':
        model_name = 'gpt-4-1106-preview'
    
    def parse_hint_selection(rawstr:str)-> tuple:
        rawstr = rawstr.strip()
        try: 
            hint_w_header, reasoning_method = rawstr.split('Promising Method: ')
            # print(reasoning_method)
            hint = hint_w_header.strip()
            reasoning_method = reasoning_method.strip().strip('`')
            for m in ['p2c', 'cot', 'pal']:
                if m in reasoning_method:
                    reasoning_method = m
                    return hint, reasoning_method
            lowered = reasoning_method.lower().replace("-"," ").replace("_", " ")
            for m_, long in zip(['cot', 'pal', 'p2c'], ['chain of thought', 'program aided language modeling', 'plan to code']):
                if long in lowered:
                    reasoning_method = m_
                    return hint, reasoning_method
            assert reasoning_method in ['cot', 'p2c', 'pal']
        except:
            hint = rawstr
            reasoning_method = 'parsing failed'
        return hint, reasoning_method
        

    # prep prompt
    prompt_tmp = PromptStr(open(prompt_f).read().strip())
    prompt = prompt_tmp.sub('QUESTION', data['question'])
    assert isinstance(prompt, str)
    messages = [
        {'role':'user', 'content': prompt}
    ]
    hint_n_select = completion_with_backoff(
            api_key=key,
            model=model_name,
            max_tokens=60, # when verbose, if 100, reaches 4130 tokens > 4097
            stop='Answer: ', # just in case the lm continues generation
            messages=messages,
            temperature=0.,
            top_p=1.0,
            n=1)['choices'][0]['message']['content'] # str
    hint, reasoning_method = parse_hint_selection(hint_n_select)
    # print(reasoning_method)
    # print(hint)
    # print(prompt)
    # print(hint)
    # print(reasoning_method)
    return hint, reasoning_method, messages

def separate_plan_code(rawstr:str)->tuple:
    # used for 5_cohlike_prompt
    # p2c results in plan\ncode so split it.
    rawstr = rawstr.strip()
    lines = rawstr.split("\n")
    found_code = False
    for i,l in enumerate(lines):
        if l.startswith('def ') and l.strip().endswith(':'):
            found_code = True
            break
    if found_code:
        plan = "\n".join(lines[:i])
        code = "\n".join(lines[i:])
    else:
        plan, code = None, None
    return plan, code

    

def parse_method(methodstr:str)->str:
    # works for 5_cohlike_prompt
    if '(PAL)' in methodstr or 'Program aided Language Model' in methodstr.replace('-', ' '):
        return 'pal'
    elif '(CoT)' in methodstr or 'Chain of Thought' in methodstr.replace('-', ' '):   
        return 'cot'
    elif '(P2C)' in methodstr or 'Plan to Code' in methodstr.replace('-', ' '):
        return 'p2c'
    else:
        return None

def query_enhanced_coh(data: dict, 
                          prompt_f: str, 
                          key: str, 
                          backbone: str,
                          n_fewshot:int=8,
                          turn_based_coh:bool=False) -> OrderedDict[str,str]:
    # 5_cohlike_prompt.txt
    # 6-shot default maximum is used


    if backbone == 'chatgpt':
        model_name = 'gpt-3.5-turbo-16k' # if n_fewshot>=5 else 'gpt-3.5-turbo'  # this prompt is kind of lengthy
    elif backbone == 'gpt4':
        model_name = 'gpt-4'
    elif backbone == 'gpt4turbo':
        model_name = 'gpt-4-1106-preview'

    def get_turn_based_coh_prompt(
                                prompt_f,
                                q:str='',
                                n_fewshot:int=8)->list:
        assert q, 'need question to be fed'
        messages = []
        # start constructing messages
        src_d = yaml.full_load(open(prompt_f))
        # system
        sys = {'role': 'system', 'content': src_d['system']} 
        messages.append(sys)
        # fewshot
        actual_n_fewshots = min(n_fewshot, len(src_d['assistant']))
        for i in range(actual_n_fewshots):
            usr1 = {'role': 'user', 'content': src_d['user'].pop(0)}
            ast1 = {'role': 'assistant', 'content': src_d['assistant'].pop(0)}
            messages.append(usr1)
            messages.append(ast1)
        # question of interest
        messages.append({'role':'user', 'content': src_d['user_tmp'].replace('[QUESTION]', q)})
        
        # assert length of the message
        assert len(messages) == 2+actual_n_fewshots*2, 'L430: get_turn_based_coh_prompt() fails'
        return messages
    
    def reduce_fewshots(rawtext:str, n_fewshot:int)->str:
        '''
        helper function for turn_based_coh=False
        '''
        chunks = rawtext.split("\n\n\n")
        fewshots = chunks[1:-1]
        random.shuffle(fewshots) # in-place operation
        fewshots = fewshots[:n_fewshot] # reduced
        # assert n_fewshot >= len(fewshots), 'give smaller n_fewshot to reduce'
        reduced = "\n\n\n".join([chunks[0]] + fewshots + [chunks[-1]]) # join
        return reduced
        
    
    def parse_raw2dict(rawqueryout:str, toparse:list=None)->OrderedDict:
        '''
        helper function for output (universal for both turn_based_coh=True|False)
        '''
        lines = rawqueryout.strip().split('\n')
        parse_d = OrderedDict()
        # gather each line's index
        keys = []
        for i, l in enumerate(lines):
            for k in toparse:
                if k in parse_d: 
                    continue
                if l.startswith(k):
                    parse_d[k] = i
                    keys.append(k)
        # indices to slice
        num = len(parse_d)
        indices = list(parse_d.values())
        parse_dd = OrderedDict.fromkeys(keys)
        # parse_dd = actual parsed dict
        for i in range(num):
            if i == num-1:
                content = lines[indices[i]:]
            else:
                content = lines[indices[i]: indices[i+1]]
            assert content, 'empty content'
            parse_dd[toparse[i]] = "\n".join(content)
        if 'Solution:' not in parse_dd.keys():
            parse_dd['Solution:'] = None
            print(f'paring failed:\n{rawqueryout}')
        else:
            try: 
                parse_dd['Solution:'] = parse_dd['Solution:'].replace('Solution:', '').strip()
                if '\nAnswer: ' in parse_dd['Solution:']:
                    parse_dd['Solution:'] = parse_dd['Solution:'].split('\nAnswer: ')[0]
            except:
                print('no Solution: found')
                pass
        return parse_dd

    # prep prompt
    if turn_based_coh: # *.yaml
        messages = get_turn_based_coh_prompt(prompt_f, 
                                             q=data['question'],
                                             n_fewshot=n_fewshot)
    else: #*.txt 
        rawprompt = open(prompt_f).read().strip()
        if 'cotpal' not in prompt_f:
            if n_fewshot > 6:
                print(f'max k_fewshot for cotpalp2c is 6 ({n_fewshot=}->6)') 
                n_fewshot = 6
            if n_fewshot>0 and n_fewshot<6:
                rawprompt = reduce_fewshots(rawprompt, n_fewshot)
        else:
            if n_fewshot > 2:
                print(f'max k_fewshot for cotpal is 2 ({n_fewshot=}->2)') 
                n_fewshot = 2
            if n_fewshot>0 and n_fewshot<2:
                rawprompt = reduce_fewshots(rawprompt, n_fewshot)
        prompt_tmp = PromptStr(rawprompt)
        prompt = prompt_tmp.sub('QUESTION', data['question'])
        assert isinstance(prompt, str)
        messages = [
            {'role':'user', 'content': prompt}
        ]
    print('T=0 for query_enhanced_coh() (manually set)')
    print('seed=777 for query_enhanced_coh() (manually set) from nov 11')
    
    if 'solvetwice' in prompt_f:
        stop_tok = "\n\n\n"
    else:
        stop_tok = "\nEvaluation: "
    raw_query_out = completion_with_backoff(
            api_key=key,
            seed=777,
            model=model_name,
            max_tokens=1024, 
            stop=stop_tok, 
            messages=messages,
            temperature=0.,
            top_p=1.0,
            n=1)['choices'][0]['message']['content'] # str
    if 'solvetwice' in prompt_f:
        toparse = ['Failed Method:', 'Failed Attempt:', 'Answer:', 'Evaluation:', 'Reflection:', 'Hint:', 'Successful Method:', 'Solution:']
    else:
        toparse = ['Failed Method:', 'Hint:', 'Successful Method:', 'Solution:', 'Answer:']
    parsed_dict = parse_raw2dict(raw_query_out, toparse = toparse)
        
    

    return parsed_dict, raw_query_out, messages


def query_math(
        data: dict, 
        key: str, 
        cot_temperature: float, 
        pal_temperature: float, 
        sc_num: int, 
        backbone: str, 
        
        plan_temperature: float=.0, # when use_plancode == True
        code_temperature: float=.0,
        k_fewshot:int=0,
        use_plancode:bool=False,
        ablation:str='',
        actor_selection_prompt:str='', # 10/14 assuming kshot harvesting is done, test actor potential
        prog_hint_prompting:bool=False, # 10/14 whether to do `hint injection`
        cohprompt:str='', # 10/19 coh exp
        when_only_conflict:int=-1, # oct26~ exp
        tgt_conflict:bool=False, # nov11 exp
        turn_based_coh:bool=False, # nov11 exp 

        dbg:bool =False,  # dbgmode

        ):
    '''
    This function is used to query OpenAI for answers in arithmetic tasks. It contains three steps:
    1. Query CoT for solutions
    2. Query PAL for solutions
    3. Query model selection answers

    Note that we only query selection answers when CoT and PAL answers are different. Otherwise, we directly use CoT or PAL answers.

    We also use majority voting to select the final answer if we have multiple self-consistency samples.

    Args:
        data: a dict containing the question and answer
        key: the OpenAI API key
        cot_temperature: the temperature used in CoT. 0 for greedy decoding. We set it to 0.5 for self-consistency samples.
        pal_temperature: the temperature used in PAL. 0 for greedy decoding. We set it to 0.8 for self-consistency samples.
        sc_num: the number of self-consistency samples
        backbone: ChatGPT or GPT-4

    Returns:
        to_dump_data: a dict containing the question, answer, the final answer and other information
    
        
    added query_plancode routine inside
    '''

    cot_answers = []
    pal_answers = []
    p2c_answers = []
    cot_solutions = []
    pal_solutions = []
    p2c_solutions = []
    selection_solutions = []
    final_answers = []

    good_solutions = []
    rawouts = []
    plan = ''
    task_prompts = [] # not an actor prompt, but a task prompt

    for i in range(sc_num):
        if when_only_conflict in [2,3]: # actor selection or coh-enhanced when only conflict.
            cot_ans = None
            pal_ans = None
            p2c_ans = None
            # selection_ans = None # this for model-selection
            final_ans = None
            if tgt_conflict:
                cot_ans = data['cot_executed'][0]
                pal_ans = data['pal_executed'][0]
                cot_solution = data['cot_generated']
                pal_solution = data['pal_generated']
                if when_only_conflict==3:
                    p2c_ans = data['p2c_executed'][0]
                    p2c_solution = data['p2c_generated']
                    
            else:
                cot_solution, query_msg = query_cot(
                    data, key, cot_temperature, backbone=backbone)
                # do cot
                if cot_solution is None:
                    print('Time out') 
                    return None
                else:
                    cot_ans = extract_num_turbo(cot_solution[0])
                    cot_answers.append(cot_ans)
                    cot_solutions.append(cot_solution[0])
                # do pal
                pal_solution, query_msg = query_pal(
                    data, key, pal_temperature, backbone=backbone)
                if pal_solution is None:
                    print('Time out')
                    return None
                else:
                    pal_ans = safe_execute_turbo(pal_solution[0]) # testing pal-generated code and getting answer from it
                    pal_answers.append(pal_ans)
                    pal_solutions.append(pal_solution[0])
                if when_only_conflict==3:
                    # do p2c 
                    p2c_solution, plans, query_msg = query_plancode(data, key=key, plan_temperature=plan_temperature, code_temperature=code_temperature, backbone=backbone, k_fewshot=k_fewshot)
                    if p2c_solution is None:
                        print('Time out')
                        return None
                    else:
                        p2c_ans = safe_execute_turbo(p2c_solution[0]) # testing p2c-generated code and getting answer from it
                        p2c_answers.append(p2c_ans)
                        p2c_solutions.append(p2c_solution[0])
                # apply --cohprompt | --actor_selection_prompt
            answers_above = [cot_ans, pal_ans]
            if when_only_conflict==3:
                answers_above.append(p2c_ans)
            
            ##########
            ## I guess this part could be hugely improvable (readability, conciseness). 
            ## `args.when_only_conflict` option should dangle just before the baseline routine or should be merged into it. but let us avoid unnecessary confusion for now...
            ##########

            if cohprompt: 
                # Does not matter when_only_conflict==2 or 3. Below works for both.
                final_ans = get_concordant_answer(answers_above)
                if final_ans is None: # (answers_above are all None) OR (not concordant) 
                    parse_dd, rawout, query_msg = query_enhanced_coh(
                                                                data, 
                                                                prompt_f=cohprompt, 
                                                                key=key, 
                                                                backbone=backbone,
                                                                n_fewshot=k_fewshot,
                                                                turn_based_coh=turn_based_coh)
                    good_solution = parse_dd['Solution:']
                    try:
                        good_method = parse_method(parse_dd['Successful Method:'])
                    except Exception as e:
                        print(e)
                        print("parse_dd['Successful Method:'] failed")
                        good_method = None
                    # start parsing
                    if good_method is None:
                        final_ans, actual_method = None, None
                    elif good_method == 'cot':
                        final_ans = extract_num_turbo(good_solution)
                        actual_method = good_method
                    elif good_method == 'pal':
                        try:
                            final_ans = safe_execute_turbo(good_solution)
                            actual_method = good_method
                        except: 
                            final_ans = extract_num_turbo(good_solution)
                            actual_method = 'cot'
                    elif good_method == 'p2c':
                        plan, code = separate_plan_code(good_solution)
                        try: 
                            final_ans = safe_execute_turbo(code)
                            actual_method = good_method
                        except: #if code is None:
                            final_ans = extract_num_turbo(good_solution)
                            actual_method = 'cot'
                    # record re-attempted answer and solutions

                    ansmap = {
                        'cot': cot_ans,
                        'pal': pal_ans,
                        # 'p2c': p2c_ans,
                    }
                    solmap = {
                        'cot': cot_solution,
                        'pal': pal_solution,
                        # 'p2c': (plan, p2c_solution),
                    }
                    if when_only_conflict == 3:
                        ansmap['p2c'] = p2c_ans
                        solmap['p2c'] = (plan, p2c_solution)
                    
                    reattempt = {'good_method': good_method, 
                                 'actual_method': actual_method, 
                                 'good_solution': good_solution} 
                    
                    try:
                        bad_method = parse_method(parse_dd['Failed Method:'])
                    except Exception as e:
                        print(e)
                        print("parse_dd['Failed Method:'] failed")
                        bad_method = None
                    reattempt['bad_method'] = bad_method
                    if 'Failed Attempt:' in parse_dd.keys():
                        bad_solution = parse_dd['Failed Attempt:']
                        reattempt['bad_solution'] = bad_solution # Failed Attempt solution considered Correct and stops generation (chatgpt) --> use that attempt
                        if final_ans is None and 'Successful Method: ' not in rawout and 'Evaluation: Correct' in rawout:
                            actual_method = f"{bad_method} (failed attempt considered `correct`)"
                            if bad_method =='p2c':
                                plan, code = separate_plan_code(bad_solution)
                                good_solution = (plan, code)
                                final_ans = safe_execute_turbo(code)
                            elif bad_method == 'pal':
                                final_ans = safe_execute_turbo(bad_solution)
                            elif bad_method == 'cot':
                                final_ans = extract_num_turbo(bad_solution)
                            else:
                                raise ValueError('failed to generate proper answer format (for both)')                    


                else:
                    assert not tgt_conflict, '--tgt_conflict cannot reach here.'
                    rawout = 'no coh (concordant)'
                    query_msg = 'no coh (concordant)'
                    good_solution = ''
                    # what was the concordant answer?
                    good_method = ''
                    actual_method = ''
                    bad_method = ''
                    for method, a in zip(['cot', 'pal', 'p2c'], answers_above):
                        if a == final_ans:
                            good_method += method
                            actual_method += method
                        else:
                            bad_method += method
                    reattempt = dict()

                # record result
                task_prompts.append(query_msg)
                rawouts.append(rawout)
                good_solutions.append(good_solution)
                
            elif actor_selection_prompt:
                if when_only_conflict==2: # oct26 experiment
                    # same as model selection but actor selection instead
                    if cot_ans is not None and pal_ans is not None:
                        # ==== Only select when CoT and PAL are different ====
                        if abs(cot_ans - pal_ans) >= 1e-3:
                            hint, reasoning_method, query_msg_actor = query_actor_selection(data,
                                                                                            prompt_f = actor_selection_prompt,
                                                                                            key= key,
                                                                                            backbone=backbone)
                            if reasoning_method == 'cot':
                                final_ans = cot_ans
                            elif reasoning_method == 'pal':
                                final_ans = pal_ans
                            else: # paring failed 
                                final_ans = cot_ans if random.random()<0.5 else pal_ans
                            selection_solutions.append(reasoning_method)
                            selection_solutions.append(hint)
                            task_prompts.append(query_msg_actor)
                        else:
                            final_ans = cot_ans
                    elif cot_ans is not None and pal_ans is None:
                        final_ans = cot_ans
                    elif cot_ans is None and pal_ans is not None:
                        final_ans = pal_ans
                    else:
                        final_ans = None
                elif when_only_conflict==3:
                    raise NotImplementedError('select from three --> need real-time flexible blurb building for actor selection')
            else:
                raise ValueError('when_only_conflict==2|3 but no --actor_selection_prompt or --cohprompt')


        else:
            if cohprompt: # coh-enhanced (10/21)
                assert not ablation
                assert not actor_selection_prompt
                parse_dd, rawout, query_msg = query_enhanced_coh(
                    data, 
                    prompt_f=cohprompt, 
                    key=key, 
                    backbone=backbone,
                    n_fewshot=k_fewshot)
                good_solution = parse_dd['Solution:']
                try:
                    good_method = parse_method(parse_dd['Successful Method:'])
                except Exception as e:
                    print(e)
                    print("parse_dd['Successful Method:'] failed")
                    good_method = None
                # start parsing
                if good_method is None:
                    final_ans, actual_method = None, None
                elif good_method == 'cot':
                    final_ans = extract_num_turbo(good_solution)
                    actual_method = good_method
                elif good_method == 'pal':
                    try:
                        final_ans = safe_execute_turbo(good_solution)
                        actual_method = good_method
                    except: 
                        final_ans = extract_num_turbo(good_solution)
                        actual_method = 'cot'
                elif good_method == 'p2c':
                    plan, code = separate_plan_code(good_solution)
                    try: 
                        final_ans = safe_execute_turbo(code)
                        actual_method = good_method
                    except: #if code is None:
                        final_ans = extract_num_turbo(good_solution)
                        actual_method = 'cot'

                
                # record result
                task_prompts.append(query_msg)
                rawouts.append(rawout)
                good_solutions.append(good_solution)
                try:
                    bad_method = parse_method(parse_dd['Failed Method:'])
                except Exception as e:
                    print(e)
                    print("parse_dd['Failed Method:'] failed")
                    bad_method = None
            elif actor_selection_prompt: # actor-selection only (10/14)
                assert not ablation, "actor_selection_prompt and ablation cannot be used together."
                hint, reasoning_method, query_msg_actor = query_actor_selection(data, 
                                                            prompt_f=actor_selection_prompt, 
                                                            key=key, 
                                                            backbone=backbone)
                if not prog_hint_prompting:
                    hint = '' #make it empty
                # reasoning_method = n/a if parsing fails
                # hint = generated string if parsing fails
                if reasoning_method == 'cot':
                    cot_solution, query_msg = query_cot(
                                        data, 
                                        key, 
                                        cot_temperature, 
                                        backbone=backbone, 
                                        hint=hint)
                    if cot_solution is None:
                        print('Time out')
                        return None
                    else:
                        cot_ans = extract_num_turbo(cot_solution[0]) # number parsed
                        cot_answers.append(cot_ans) # parsed answers stacked --> cot_executed
                        cot_solutions.append(cot_solution[0]) # unparsed answers --> cot_generated
                        final_ans = cot_ans

                elif reasoning_method == 'pal':
                    pal_solution, query_msg = query_pal(
                            data, key, pal_temperature, backbone=backbone, hint=hint)
                    if pal_solution is None:
                        print('Time out')
                        return None
                    else:
                        pal_ans = safe_execute_turbo(pal_solution[0]) # testing pal-generated code and getting answer from it
                        pal_answers.append(pal_ans)
                        pal_solutions.append(pal_solution[0])
                        final_ans = pal_ans 

                elif reasoning_method == 'p2c':
                    pal_solution, plans, query_msg = query_plancode(data, key=key, 
                                                plan_temperature=plan_temperature, 
                                                code_temperature=code_temperature, 
                                                backbone=backbone, 
                                                k_fewshot=k_fewshot, hint=hint)
                    if pal_solution is None or pal_solution == [None]:
                        print('Time out')
                        return None
                    else:
                        pal_ans = safe_execute_turbo(pal_solution[0]) # testing pal-generated code and getting answer from it
                        pal_answers.append(pal_ans)
                        pal_solutions.append(pal_solution[0])
                        final_ans = pal_ans
                else: # failed to select correctly
                    final_ans = None
                task_prompts.append({'actorquery': query_msg_actor, 'taskquery': query_msg})                
            elif ablation: # ablation != ''
                if ablation == 'cot':
                    cot_solution, query_msg = query_cot(
                                        data, 
                                        key, 
                                        cot_temperature, 
                                        backbone=backbone)
                    if cot_solution is None:
                        print('Time out')
                        return None
                    else:
                        cot_ans = extract_num_turbo(cot_solution[0]) # number parsed
                        cot_answers.append(cot_ans) # parsed answers stacked --> cot_executed
                        cot_solutions.append(cot_solution[0]) # unparsed answers --> cot_generated
                        final_ans = cot_ans

                elif ablation == 'pal':
                    pal_solution, query_msg = query_pal(
                            data, key, pal_temperature, backbone=backbone)
                    if pal_solution is None:
                        print('Time out')
                        return None
                    else:
                        pal_ans = safe_execute_turbo(pal_solution[0]) # testing pal-generated code and getting answer from it
                        pal_answers.append(pal_ans)
                        pal_solutions.append(pal_solution[0])
                        final_ans = pal_ans 

                elif ablation == 'plancode':
                    pal_solution, plans, query_msg = query_plancode(data, key=key, 
                                                plan_temperature=plan_temperature, 
                                                code_temperature=code_temperature, 
                                                backbone=backbone, 
                                                k_fewshot=k_fewshot)
                    if pal_solution is None or pal_solution == [None]:
                        print('Time out')
                        return None
                    else:
                        pal_ans = safe_execute_turbo(pal_solution[0]) # testing pal-generated code and getting answer from it
                        pal_answers.append(pal_ans)
                        pal_solutions.append(pal_solution[0])
                        final_ans = pal_ans 
                task_prompts.append(query_msg)
            else: # doing model-selection (baseline paper + p2c_v1)
                cot_ans = None
                pal_ans = None
                selection_ans = None
                final_ans = None
                cot_solution, query_msg = query_cot(
                    data, key, cot_temperature, backbone=backbone)
                if cot_solution is None:
                    print('Time out')
                    return None
                else:
                    cot_ans = extract_num_turbo(cot_solution[0])
                    cot_answers.append(cot_ans)
                    cot_solutions.append(cot_solution[0])
                if use_plancode:
                    pal_solution, plans, query_msg = query_plancode(data, key=key, plan_temperature=plan_temperature, code_temperature=code_temperature, backbone=backbone, k_fewshot=k_fewshot)
                else:
                    pal_solution, query_msg = query_pal(
                        data, key, pal_temperature, backbone=backbone)
                if pal_solution is None:
                    print('Time out')
                    return None
                else:
                    pal_ans = safe_execute_turbo(pal_solution[0]) # testing pal-generated code and getting answer from it
                    pal_answers.append(pal_ans)
                    pal_solutions.append(pal_solution[0])

                if cot_ans is not None and pal_ans is not None:
                    # ==== Only select when CoT and PAL are different ====
                    if abs(cot_ans - pal_ans) > 1e-3:
                        selection_ans = query_selection(
                            data, key, cot_solution=cot_solution, pal_solution=pal_solution, backbone=backbone)
                        if selection_ans is None:
                            print('Time out')
                            return None
                        else:
                            selection_choice = extract_choice_turbo(selection_ans[0])
                            selection_solutions.append(selection_ans[0])
                            if selection_choice == '(A)':
                                final_ans = cot_ans
                            elif selection_choice == '(B)':
                                final_ans = pal_ans
                    else:
                        final_ans = cot_ans

                elif cot_ans is not None and pal_ans is None:
                    final_ans = cot_ans
                elif cot_ans is None and pal_ans is not None:
                    final_ans = pal_ans
                else:
                    final_ans = None
                task_prompts.append(query_msg)


        final_answers.append(final_ans)

        count = Counter(final_answers)
        majority_ans = count.most_common(1)[0][0]

    # === dump data ===
    to_dump_data = OrderedDict(
        {'index': data['index'], 'question': data['question'], 'answer': data['answer'],
         'majority_ans': majority_ans, 'final_answers': final_answers,
         'cot_executed': cot_answers, 'pal_executed': pal_answers,
         'cot_generated': cot_solutions, 'pal_generated': pal_solutions,
         'choice_solution': selection_solutions,
         'iscorrect': majority_ans==data['answer'] }
    )
    if when_only_conflict in [2,3]:
        to_dump_data['p2c_generated'] = p2c_solutions
        to_dump_data['p2c_executed'] = p2c_answers
        to_dump_data['when_only_conflict'] = when_only_conflict
        if cohprompt:
            to_dump_data['good==actual'] = good_method == actual_method
            to_dump_data['good_solution'] = good_solution
            to_dump_data['bad_method'] = bad_method
            to_dump_data['good_method'] = good_method
            to_dump_data['plan'] = [plan]
            to_dump_data['rawouts'] = rawouts
            to_dump_data['actual_method'] = actual_method
            to_dump_data['reattempt'] = reattempt
    elif ablation == 'plancode':
        to_dump_data['plan'] = plans
        to_dump_data['reasoning_method'] = 'p2c'
    elif actor_selection_prompt:
        to_dump_data['hint'] = hint
        to_dump_data['prog_hint_prompting'] = prog_hint_prompting
        to_dump_data['reasoning_method'] = reasoning_method
        if reasoning_method == 'p2c':
            to_dump_data['plan'] = plans
    elif cohprompt:
        to_dump_data['good==actual'] = good_method == actual_method
        to_dump_data['good_solution'] = good_solution
        to_dump_data['bad_method'] = bad_method
        to_dump_data['good_method'] = good_method
        to_dump_data['plan'] = [plan]
        to_dump_data['rawouts'] = rawouts
        to_dump_data['actual_method'] = actual_method
    
    to_dump_data['task_prompts'] = task_prompts

    return to_dump_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--dataset', type=str, choices=['dbg', 
                        'gsm8k', 'svamp', 'asdiv', 'singleeq', 'singleop',
                        'singleaddsub', 'multiarith'], default='gsm8k')
    parser.add_argument('--backbone', type=str,
                        choices=['chatgpt', 'gpt4'], default='gpt4')
    parser.add_argument('--cot_temperature', type=float, default=0.)
    parser.add_argument('--pal_temperature', type=float, default=0.)
    parser.add_argument('--sc_num', type=int, default=1,
                        help='Self-consistency samples. 1 indicates greedy decoding')
    parser.add_argument('--output_dir', type=str, default='../output/')
    # parser.add_argument(
        # '--key', type=str, default='sk-', required=True)

    # plancode options
    parser.add_argument('--use_plancode', action='store_true')
    parser.add_argument('--plan_temperature', type=float, default=0.)
    parser.add_argument('--code_temperature', type=float, default=0.)
    parser.add_argument('--k_fewshot', type=int, default=8) #  >= 0
    # ablative options
    parser.add_argument('--ablation', type=str, default='', choices=['cot', 'plancode', 'pal'], 
                        help='for ablation study: use only one method to reason on gsm8k')
    
    # actor_prompt_test
    parser.add_argument('--actor_selection_prompt', type=str, default='', help='this flag will run actor selection prompt test.')
    parser.add_argument('--prog_hint_prompting', action='store_true', help='this flag will run prog_hint_prompting test: prepend hint when querying the solution')

    # cohprompt (coh exp)
    parser.add_argument('--cohprompt', type=str, default='', help='path to customprompt file')

    # 10/26, 27... only conflict
    parser.add_argument('--when_only_conflict', type=int, help='selecting from 2 or 3', default=-1)
    '''
        1. randomly sample amongst methods at first (i.e. cot pal plancode)
        2. verify the answer with the prompt (I guess this will be a critic?)
        3. if considered fail in `2`, sample a method again and retry (prompted to sample method with the question), the method prompt will be augmented with the wrong answer
        4. repeat `2` and `3` until the answer is correct or the number of cycle exceeds `n_cycle`
    '''

    # 11/11 targetting only conflict
    '''
    our algorithm applies only when the results from three methods are disconcordant.
    thus apply those only on conflict cases to test urgently 
    '''
    parser.add_argument('--tgt_conflict', action='store_true', help='this flag will allow running algorithms only on conflict cases.')
    parser.add_argument('--dbg', action='store_true', help='this flag will disable retrying logics to catch bugs')


    args = parser.parse_args()

    key = open('../openai_key.txt').read().strip()

    # prep experiements-common
    start_index = args.start
    end_index = args.end
    dataset_name = args.dataset
    cot_temperature = args.cot_temperature
    pal_temperature = args.pal_temperature
    backbone = args.backbone
    sc_num = args.sc_num
    output_dir = args.output_dir

    start_time_0 = time.time()
    print('Current time: ', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime()))

    dt_string = datetime.now().strftime("%m_%d_%H_%M")

    if dataset_name == 'gsm8k':
        dataset = jsonlines_load('../dataset/gsm8K_test.jsonl')
    elif dataset_name == 'dbg':
        dataset = jsonlines_load('../dataset/dbg.jsonl') # 2 lines of gsm8k test
    elif dataset_name == 'svamp':
        dataset = jsonlines_load('../dataset/svamp.jsonl')
    elif dataset_name == 'asdiv':
        dataset = jsonlines_load('../dataset/asdiv.jsonl')
    elif dataset_name == 'singleeq':
        dataset = jsonlines_load('../dataset/single_eq.jsonl')
    elif dataset_name == 'singleop':
        dataset = jsonlines_load('../dataset/single_op.jsonl')
    elif dataset_name == 'singleaddsub':
        dataset = jsonlines_load('../dataset/single_addsub.jsonl')
    elif dataset_name == 'multiarith':
        dataset = jsonlines_load('../dataset/multiarith.jsonl')
    
    if args.tgt_conflict:
        cotpal_conflict_jsls = list(Path('../output/nov11_tgt_conflict').glob(f'**/coh_cotpal_1/conflict*.jsonl'))
        if 'cotpal' in args.cohprompt:
            conflict_jsls = cotpal_conflict_jsls        
        else:
            conflict_jsls = list(Path('../output/nov11_tgt_conflict').glob(f'**/conflict*.jsonl'))
            # conflict_jsls = list(Path('../output/nov12_later_or_donot/baseline/').glob(f'**/conflict*.jsonl'))
            print(f"{conflict_jsls=}")
            conflict_jsls = [p for p in conflict_jsls if p not in cotpal_conflict_jsls]
        datasets_backbones_paths = [(jsonlines_load(jslf), jslf.parent.parent.name, jslf) for jslf in conflict_jsls]
        datasets = [e[0] for e in datasets_backbones_paths]
        backbones = [e[1] for e in datasets_backbones_paths]
        paths = [e[2] for e in datasets_backbones_paths]
    else: # only one dataset
        datasets = [dataset]

    # === slice data based on start and end ===
    for ii, dataset in enumerate(datasets): 
        total_num = len(dataset)
        print('total data: ', total_num)
        unfinished_tasks = []
        if args.tgt_conflict: # conflict case only run
            
            tasks = dataset 
            if args.dbg:
                tasks = dataset[:10]
            task_num = len(tasks)
            print('Current total tasks: ', task_num)

            assert args.cohprompt, f'when --tgt_conflict, need --cohprompt {args.cohprompt}'
            # adjust the following arguments for `tgt_conflict==True` setting 
            datasetpath = paths[ii]
            args.when_only_conflict = 3 if datasetpath.parent=='coh' else 2 # 3 for 3model-coh 2 for 2model-coh
            backbone = backbones[ii]
            output_path = paths[ii].parent/Path(args.cohprompt).stem
            
            turn_based_coh = False
            if args.cohprompt.endswith('.yaml'):
                turn_based_coh = True
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            args.output_dir = output_path
            if args.dbg:
                save_path = output_path/f"dbg_tgt_conflict_{dt_string}_{datasetpath.name}"
            else:
                save_path = output_path/f"tgt_conflict_{dt_string}_{datasetpath.name}"
            

            print(f"{datasetpath=}")
            print(f'{args.tgt_conflict=}')
            print(f"running with --tgt_conflict will override --output_dir")
            print(f"\t{args.when_only_conflict=}")
            print(f"\t{backbone=}")
            print(f"\t{args.cohprompt=}")
            print(f"\t{turn_based_coh=}")
            print(f"\t{args.output_dir=}")
            print(f"\t{save_path=}")

        
        else: # normal run
            if end_index == -1:
                end_index = total_num

            if end_index > total_num:
                end_index = total_num

            tasks = dataset[start_index:end_index]
            task_num = len(tasks)
            print('Current total tasks: ', task_num)

            
            output_path = os.path.join(output_dir, f'{backbone}/')
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            save_path = os.path.join(output_path,
                                    f'{dataset_name}_k{args.k_fewshot}_sc{sc_num}_s{start_index}_e{end_index}_{dt_string}.jsonl')
            print(save_path)

        # === run experiments ===
        progress_bar = tqdm(range(task_num))
        for i in range(task_num):
            task = tasks[i]
            start_time = time.time()
            count = 0
            
            if dataset_name == 'dbg' or args.dbg:
                ans = query_math(
                        task, key=key, cot_temperature=cot_temperature,
                        pal_temperature=pal_temperature, sc_num=sc_num,backbone=backbone,
                        plan_temperature=args.plan_temperature,
                        code_temperature=args.code_temperature,
                        k_fewshot=args.k_fewshot,
                        use_plancode=args.use_plancode, # for model selection experiment
                        ablation=args.ablation, # for onlyone method ablation study
                        actor_selection_prompt=args.actor_selection_prompt, # for actor selection prompt test
                        prog_hint_prompting=args.prog_hint_prompting, # whether to inject hint to query solution
                        cohprompt=args.cohprompt, # for cohprompt exps,
                        when_only_conflict=args.when_only_conflict, 
                        tgt_conflict = args.tgt_conflict,
                        turn_based_coh = turn_based_coh,
                        dbg = args.dbg
                        )
                                

                progress_bar.update(1)
                if ans is not None:
                    with open(save_path, "a+") as fout:
                        fout.write(json.dumps(ans)+'\n')

            else:
                while True:
                    try:
                        count += 1

                        # if args.cohprompt.endswith('prompts/prep_reflexion/5_cohlike_prompt.txt') and args.k_fewshot>6: # oct19 exp
                        #     args.k_fewshot = 6
                        #     print('for 5_cohlike_prompt.txt, k_fewshot is maximum at 6')
                            

                        ans = query_math(
                            task, key=key, cot_temperature=cot_temperature,
                            pal_temperature=pal_temperature, sc_num=sc_num,backbone=backbone,
                            plan_temperature=args.plan_temperature,
                            code_temperature=args.code_temperature,
                            k_fewshot=args.k_fewshot,
                            use_plancode=args.use_plancode, # for model selection experiment
                            ablation=args.ablation, # for onlyone method ablation study
                            actor_selection_prompt=args.actor_selection_prompt, # for actor selection prompt test
                            prog_hint_prompting=args.prog_hint_prompting, # whether to inject hint to query solution
                            cohprompt=args.cohprompt, # for custom prompt experiment,
                            when_only_conflict=args.when_only_conflict, 
                            tgt_conflict = args.tgt_conflict,
                            turn_based_coh = turn_based_coh,
                            dbg= args.dbg,
                            )

                                    
                    except Exception as e:
                        print(e)
                        ans = None
                    if ans is not None:
                        with open(save_path, "a+") as fout:
                            fout.write(json.dumps(ans)+'\n')
                        progress_bar.update(1)
                        break
                    else: 
                        if count>1:
                            print(f'tried {count} times, passing')
                            print('Current Task: ', i)
                            unfinished_tasks.append(task)
                            count=0
                            break
                        else:
                            print("retrying (main)")
                            time.sleep(random.uniform(1,3))
            
                

        end_time_0 = time.time()
        print('Finish at time: ', time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime()))
        print(f'Time used: {end_time_0 - start_time_0} seconds')
        if len(unfinished_tasks) > 0:
            unfinished_path = Path(save_path).parent/'unfinished'/Path(save_path).name.replace('.jsonl', '_unfinished.jsonl')
            if not unfinished_path.parent.exists():
                unfinished_path.parent.mkdir(exist_ok=True, parents=True)
            unfinished_path = str(unfinished_path)
            with jsl.open(unfinished_path, 'w') as writer:
                writer.write_all(unfinished_tasks)
                print(f'Unfinished tasks at: \n\t{unfinished_path}')
            with open(f'{unfinished_path}.args', 'w') as f:
                print(args, file=f)
            print(f'Unfinished args at: \n\t{unfinished_path}.args')

            

        print('Done')
