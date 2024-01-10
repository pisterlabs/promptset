import yaml 
from typing import Sequence, Mapping, Any, Union, Callable
PLAN_F = '/Users/seonils/dev/llm-reasoners/examples/Model-Selection-Reasoning/src/prompts/prompts_plan_v2.yaml'
CODE_F = '/Users/seonils/dev/llm-reasoners/examples/Model-Selection-Reasoning/src/prompts/prompts_code_v2.yaml'
import openai
KEY = open('/Users/seonils/dev/llm-reasoners/examples/Model-Selection-Reasoning/openai_key.txt').read().strip()
openai.api_key = KEY # set key
PLAN_PROMPTS_D= yaml.full_load(open(PLAN_F))
CODE_PROMPTS_D = yaml.full_load(open(CODE_F))


def get_plan_prompt(data: dict, k_fewshot:int=0, hint:str='')->str:
    '''
    prep prompt for plan generation
    '''
    prompt_d = PLAN_PROMPTS_D
    
    q = data['question'] 
    system = prompt_d['system_msg']
    user_tmp = prompt_d['user_template'] 
    if hint:
        user_attempt = user_tmp.replace('{QUESTION}', f"Question: {q} ({hint})")
    else:
        user_attempt = user_tmp.replace('{QUESTION}', f"Question: {q}")

    # print(user_attempt)
    fewshots_user = prompt_d['fewshots_user'][:k_fewshot] # list of fewshot strings include Question: as a stop sequence.
    fewshots_assistant = prompt_d['fewshots_assistant'][:k_fewshot]
    
        
    msgs =  [{'role': 'system', 'content': system},]
    for fu, fa in zip(fewshots_user, fewshots_assistant):
        usr = {'role': 'user', 'content': fu}
        astnt = {'role': 'assistant', 'content': fa}
        msgs.append(usr)
        msgs.append(astnt)
    msgs.append({'role':'user', 'content': user_attempt})

    return msgs
    
    
def get_plan2code_prompt(data:dict, plan:str='', k_fewshot:int=0, custom_idxs:list=None, hint:str=''):
    # little bit revision from PAL prompt.
    # `solution()` is returned (can execute with solution() call w/o argument
    prompt_d = CODE_PROMPTS_D
    
    q = data['question'] 
    system = prompt_d['system_msg']
    user_tmp = prompt_d['user_template'] 
    if hint:
        q = f"{q} ({hint})"
    user_attempt = user_tmp.replace('{PLAN}', plan).replace('{QUESTION}', f"Question: {q}")
    # print(q)

    if not custom_idxs:
        fewshots_user = prompt_d['fewshots_user'][:k_fewshot] # list of fewshot strings include Question: as a stop sequence.
        fewshots_assistant = prompt_d['fewshots_assistant'][:k_fewshot]
    else:
        fewshots_user = [prompt_d['fewshots_user'][i] for i in custom_idxs]
        fewshots_assistant = [prompt_d['fewshots_assistant'][i] for i in custom_idxs]
        
    msgs =  [{'role': 'system', 'content': system},]
    for fu, fa in zip(fewshots_user, fewshots_assistant):
        usr = {'role': 'user', 'content': fu}
        astnt = {'role': 'assistant', 'content': fa}
        msgs.append(usr)
        msgs.append(astnt)
    msgs.append({'role':'user', 'content': user_attempt})
    
    return msgs


def postprocess_plan(rawanswer:str):
    # lines = [l for l in rawanswer.split('\n') if '</end>' not in l]
    lines = rawanswer.split('\n')
    if len(lines)>=1:
        plan_ = "\n".join(lines)
    else:
        print('plan gen failed')
        print(f"{rawanswer=}")
        plan_ = ''
    return plan_

# def postprocess_code_answer(rawanswer:str, docdef:str='', k_fewshot:int=0):
def postprocess_code(rawanswer:str, k_fewshot:int=0):
    try:
        # 1 removing starting wrap ```
        if "```python" in rawanswer:
            code = rawanswer.split("```python")[-1]
        elif rawanswer.startswith('```'):
            rawanswer = rawanswer.split('```')[-1]
        
        # 2 removing ``` at the end
        code = rawanswer.split("```")[0] #ending ``` removal
        
        # in v1, I tried force decode in prompt which caused so many errors, I will not do it here.
        # if k_fewshot>0: # just use output do not modif
        #     if code.startswith('def solution():'):
        #         pass
        #     else:
        #         code = docdef + '\n' + (code if code.startswith('\t') else f"\t{code}")
        code = remove_prints(code)
        assert code
        # exec(code) # check if it is executable # this is done in tool.py:safe_execute_turbo
    except: 
        print('code gen fails (unexecutable or funcname?)')
        print(f"code:\n{rawanswer}")
        code = ''
    return code

def remove_prints(code:str)->str:
    lines = code.split("\n")
    lines_ = [l if not l.startswith('print(') else l.replace('print(', '# print(') for l in lines]
    code_ = "\n".join(lines_)
    return code_

 
def kvprint(record):
    for r in record:
        print(r['role'])
        print(r['content'])


if __name__ == '__main__':
    prompt_dd = yaml.full_load(open('prompts_code.yaml'))
    print()

    data = {"question": "Guesstimate the how many times more the ladies' toilet needed to make the same line length for both gender (it is well-known that the line for the ladies' are much lengthier)?"}
    pp2 = get_plan_prompt(data, k_fewshot=8)
    pp2r = openai.ChatCompletion.create(messages = pp2, model='gpt-3.5-turbo', stop='Question:')['choices'][0]['message']['content']
    print(f"{pp2r=}")   

    cp2 = get_plan2code_prompt(data, plan=pp2r, k_fewshot=8)
    print("===========")
    kvprint(pp2)
    print("===========")
    kvprint(cp2)
    print("===========")
    cp2r = openai.ChatCompletion.create(messages = cp2, model = 'gpt-3.5-turbo', stop= 'Question:')['choices'][0]['message']['content']
   
    print(cp2r)

    ''' # I'm satisfied with the test result below.

    Question: Guesstimate the how many times more the ladies' toilet needed to make the same line length for both gender (it is well-known that the line for the ladies' are much lengthier)?

    Guide:
    1. Start by estimating the length of the line for the ladies' toilet.
    2. Estimate the length of the line for the men's toilet.
    3. Divide the length of the line for the ladies' toilet by the length of the line for the men's toilet.
    4. Round the result to the nearest whole number.
    5. Return the rounded result as the number of times more the ladies' toilet needed to make the same line length for both genders.

    ===========
    def solution(ladies_line_length, mens_line_length):
        times_more = round(ladies_line_length / mens_line_length)
        return times_more
    '''