from llama_cpp import Llama
import json 
import openai
import numpy as np 
from utils import * 

def create_prompt(f,x_names,x,u,data,subject,objective_description,prev_justifications):
    prompt =  " You are an expert in " + subject + " You have been the following context:\n"
    objective = objective_description
    prompt += objective + '\n'

    prompt += " You must select the best solution to achieve this goal from a set of " + str(len(x)) + " alternatives."
    prompt += " The true objective function for each solution is unknown."
    prompt += " Each solution is associated with a utility value, which quantifies how much the computer believes the solution is the best, this is different from the underlying objective, which is unknown"
    prompt += " The utility values are calculated with no regard to the physical meaning of the solutions."
    prompt += " You must condition these values with your own expertise in " + subject + " which will inform your final decision."
    prompt += " If the utility values are the same or similar, then you MUST take into account the physical meaning of the specific variables."
    prompt += " Do not just select a solution solely based on the utility value, you absolutely must think step-by-step and consider additional knowledge.\n"


    round = 3
    for i in range(len(x)):
        for j in range(len(x[i])):
            x[i][j] = np.round(x[i][j],round)

    u_ind = np.argsort(-np.array(u))
    order_endings = ['st','nd','rd','th']
    # order_endings = [order_endings[i] for i in range(len(u_ind))]

    for i in range(len(x)):
        sol_str = ''.join([x_names[j]+': '+ str(np.round(x[i][j],round)) +', ' for j in range(len(x[i]))])
        best_ind = u_ind[i]
        prompt += f'''Solution {str(i+1)}: {sol_str}, Utility value, U(x) = {u[i]} ({best_ind+1}{order_endings[best_ind]} best)\n'''
    
    prev_data_len = len(data['previous_iterations'])

    prompt += f'''
    Below is a JSON object containing the previous {str(prev_data_len)} iterations of the optimisation process, the inputs are respective to the variables described above, and the outputs are the objective function values.
    You are welcome to use this information in an attempt to infer which of the alternative solutions will have the highest objective function.\n
    '''
    if prev_justifications == True:
        prompt += '''
        This may include your previous justifiction given for selecting a datapoint. Consider previous justifications, they may be wrong, but they may also be correct.
        '''
    clean_data = []
    for i in range(prev_data_len):
        x_clean = {}
        for j in range(len(data['previous_iterations'][i]['inputs'])):
            x_val = (data['previous_iterations'][i]['inputs'][j])
            x_val = np.round(x_val,round)
            try:
                x_clean[x_names[j]] = x_val.item()
            except:
                x_clean[x_names[j]] = x_val

        clean_data.append({'inputs':x_clean,'objective':np.round(data['previous_iterations'][i]['objective'],round)})
        if prev_justifications == True:
            try:
                clean_data[i]['reason'] = data['previous_iterations'][i]['reason']
            except:
                continue
    data = {'previous_iterations':clean_data}
    prompt += "\n" + json.dumps(data) + '\n'

    prompt += '''
    Provide your response ONLY as a JSON object containing the key "choice" and the key "reason"
    "choice": the index of the solution you believe is the best (1-indexed)
    "reason": a minimum 10 and maximum 30 word explanation of your reasoning for selecting the solution indexed in "choice".

    Reasoning Rules:
    The reasoning CANNOT contain reference to your status as an expert, it must start from the reason itself.
    The reasoning MUST be an appropriate and concise size.
    The reasoning CANNOT reference the objective function of a solution, because this information is unknown.
    The reasoning CANNOT reference ONLY the utility function value but you SHOULD use relative utility values for justification.
    You must make specific reference to the individual variables and their respective values.

    JSON: 
    '''
    return prompt

def run_prompt(llm,prompt):
    if llm.__class__ != str: 
        llm.reset()
        res = llm(prompt, max_tokens=1028,temperature = 0.1,stop=['}'],echo=False)
        res = res['choices'][0]['text']
        res = post_process_local(res)

    else:
        messages=[
                {"role": "user", "content": prompt},
            ]

        with open("misc/api_key.txt") as f:
            openai.api_key = f.readline()

        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo-0613',
            messages=messages,
            temperature=0.3
        )
        response_message = response["choices"][0]["message"]['content']
        res = post_process_remote(response_message)
    return res 



def post_process_local(res):
    print(res)
    res = ''.join(res)
    res = res.replace('\n','')
    res = res.replace('\t','')
    res = '{'+ res.split('{')[-1] + '}'
    try:
        res = json.loads(res)
        return res
    except:
        print('Invalid JSON, failing to standard BO....')
        return {'choice': 'NaN','reason': 'NaN'}


def post_process_remote(res):
    res = json.loads(res)
    return res