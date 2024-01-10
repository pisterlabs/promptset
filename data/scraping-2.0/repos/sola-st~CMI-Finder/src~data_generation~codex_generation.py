import os
import openai
import libcst as cst
import time

openai.api_key = "REPLACE THIS WITH YOUR KEY"

def get_wrong_consistent(message):
    
    message = message.replace('\n', '')
    message = " ".join(message.split())
    message = 'raise ' + message+ ' if'
    
    response = openai.Completion.create(
    engine="davinci-codex",
    prompt= message,
    temperature=0,
    max_tokens=64,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    completion = response['choices'][0]['text']
    if '\n' in completion:
        condition = completion.split('\n')[0]
    else:
        condition = completion
    if 'else' in condition:
        condition = condition.split('else')[0]
    return condition 

def complete_from_condition(cond):
    response = openai.Completion.create(
    engine="davinci-codex",
    prompt= cond,
    temperature=0,
    max_tokens=64,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    completion = response['choices'][0]['text']
    if '\n' in completion:
        return completion.split('\n')[0]
    else:
        return completion

def complete_from_message(message):
    
    response = openai.Completion.create(
    engine="davinci-codex",
    prompt= message,
    temperature=0,
    max_tokens=64,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    completion = response['choices'][0]['text']
    if '\n' in completion:
        condition = completion.split('\n')[0]
    else:
        condition = completion
    if 'else' in condition:
        condition = condition.split('else')[0]
    return condition 


def generate_inconsistent(target_statements, cool_down=60, api_key="NO API KEY"):
    codex_comp = []
    for inc in target_statements:
        d, cond, message = inc
        try:
            tree = cst.parse_statement(d)
        except:
            continue
            
        if_test = cst.Module([tree.test]).code
        
        if cond != if_test:
            raise_ind = d.find('raise')
            open_p = d[raise_ind:].find('(')
            left = 'if '+ cond + ' : ' + d[raise_ind:raise_ind+open_p+1]
            try:
                comp = complete_from_condition(left)
            except Exception as e:
                print(e)
                print(len(codex_comp))
                time.sleep(cool_down)
                comp = complete_from_condition(left)
            codex_comp.append((d, left, comp))
        elif message not in d:
            left = message + ' if'
            try:
                comp = complete_from_message(left)
            except:
                print(len(codex_comp))
                time.sleep(cool_down)
                comp = complete_from_message(left)
            codex_comp.append((d, left, comp))

    return codex_comp