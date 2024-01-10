# %%
import json 
import pickle
import openai 
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

openai.api_key_path = openai.api_key_path = '/Users/seacow/School/UPenn/Research/Procedural Reasoning/api_keys/harry_api_key.txt'


def get_strategy_qa_response(type='more'):
    res_dict = {}
    for key, dict in tqdm(data.items()):
        temp_dict = {}
        question_set = []
        steps = dict['steps']
        temp_dict = {'goal': dict['goal'], 'steps': steps}
        for step in steps:
            for content in step:
                if content.get('type') == 'multihop':
                    cur_q = f"Is it {type} likely that {content['event'].lower().replace('.', '')}?"
                    if cur_q not in question_set:
                        question_set.append(cur_q)

        context_bank = []
        prompt = ''
        for step in steps[1:]:
            for content in step:
                if content.get('type') == 'step':
                    prompt += content['step'].strip() + ' '
                    context_bank.append(prompt)
                    
        proc_res = []
        for i, q in enumerate(question_set):
            q_res = []
            for context in context_bank:
                cur_prompt = f"Context: {context}\nQuestion:{q}\nTake it step by step:\n"
                completion = gpt3_followup(cur_prompt)
                cur_prompt += completion
                cur_prompt += "Therefore, the answer to the original question is"
                print(cur_prompt)
                pred = gpt3_final(cur_prompt)
                q_res.append(parse_completion(pred, type))
            proc_res.append(q_res)
        temp_dict['responses'] = proc_res
        res_dict[key] = temp_dict
    return res_dict


def parse_completion(pred, type):
    if 'true' in pred.lower():
        out = type
    else:
        out = f'not {type}'
    return out


def gpt3_followup(prompt):
    ind = 1
    while ind:
        try:
            ret = openai.Completion.create(
                engine='curie:ft-ccb-lab-members:strategyqa-2022-10-08-20-03-23',
                prompt=prompt,
                temperature=0,#0.7
                max_tokens=256,
                top_p=1,
                logprobs=5,
                frequency_penalty=0,
                presence_penalty=0,
                stop=['Therefore']
            )
            ret = ret["choices"][0]['text']
            ind = 0
        except:
            pass
    return ret
    

def gpt3_final(prompt):
    ind = 1
    while ind:
        try:
            ret = openai.Completion.create(
                engine='curie:ft-ccb-lab-members:strategyqa-2022-10-08-20-03-23',
                prompt=prompt,
                temperature=0,#0.7
                max_tokens=256,
                top_p=1,
                logprobs=5,
                frequency_penalty=0,
                presence_penalty=0,
                stop=['\n']
            )
            ret = ret["choices"][0]['text']
            ind = 0
        except:
            pass
    return ret

#%%
with open('/Users/seacow/School/UPenn/Research/Procedural Reasoning/v2/data/data_test_v2.json', 'r') as f:
    data = json.load(f)
f.close()
#%%
res_more_likely = get_strategy_qa_response(type='more')

#%%
res_less_likely = get_strategy_qa_response(type='less')

# %%
#with open('/Users/seacow/School/UPenn/Research/Procedural Reasoning/v2/strategyqa/results/res_more_likely.pkl', 'wb')as f:
#    pickle.dump(res_more_likely, f)
#f.close()
#
#with open('/Users/seacow/School/UPenn/Research/Procedural Reasoning/v2/strategyqa/results/res_less_likely.pkl', 'wb')as f:
#    pickle.dump(res_more_likely, f)
#f.close()
##
##%%
#with open('/Users/seacow/School/UPenn/Research/Procedural Reasoning/v2/strategyqa/results/res_more_likely.pkl', 'rb') as f:
#    res_more_likely = pickle.load(f)
#f.close()
#
#with open('/Users/seacow/School/UPenn/Research/Procedural Reasoning/v2/strategyqa/results/res_less_likely.pkl', 'rb') as f:
#    res_less_likely = pickle.load(f)
#f.close()

# %%
gold_event_change = []
for dict in res_more_likely.values():
    events = []
    visited = []
    event_names = {}
    idx = 0
    for step_lst in dict['steps']:
        cur_state = []
        for step in step_lst:
            if step.get('type') == 'multihop':
                if (cur_event := step['event']) not in visited:
                    visited.append(cur_event)
                    event_names[cur_event] = f"event{idx}"
                    idx += 1
                if 'more' in step['change']:
                    cur_state.append((event_names[cur_event], 'more'))
                else:
                    cur_state.append((event_names[cur_event], 'less'))
        events.append(cur_state)
    gold_event_change.append(events)
                

# %%
all_more_likely = []
for dict in res_more_likely.values():
    predict_more_events = [[] for _ in range(len(dict['steps']))]
    for i, res_lst in enumerate(dict['responses']):
        for j, res in enumerate(res_lst):
            if 'not' not in res.lower():
                predict_more_events[j].append((f'event{i}', 'more'))
    all_more_likely.append(predict_more_events)

#%%
all_less_likely = []
for dict in res_less_likely.values():
    predict_less_events = [[] for _ in range(len(dict['steps']))]
    for i, res_lst in enumerate(dict['responses']):
        for j, res in enumerate(res_lst):
            if 'not' not in res.lower():
                predict_less_events[j].append((f'event{i}', 'less'))
    all_less_likely.append(predict_less_events)

#%%
final_res = []
for (predict_more_events, predict_less_events) in zip(all_more_likely, all_less_likely):
    cur_result = [[] for _ in range(len(predict_more_events))]
    for i, (res1, res2) in enumerate(zip(predict_more_events, predict_less_events)):
        event_in_res1 = [t[0] for t in res1]
        event_in_res2 = [t[0] for t in res2]

        for t in res1:
            if t[0] not in event_in_res2:
                cur_result[i].append(t)
        
        for t in res2:
            if t[0] not in event_in_res1:
                cur_result[i].append(t)
    final_res.append(cur_result)
    

y_true = []
y_pred = []
for gt, pred in zip(gold_event_change, final_res):
    events = []
    for lst in gt:
        if lst:
            for t in lst:
                cur_event = t[0]
                if cur_event not in events:
                    events.append(cur_event)
    
    for event in events:
        gold_change = 0
        for lst in gt:
            for t in lst:
                if t[0] in events:
                    if t[1] == 'more':
                        gold_change = 1
                    else:
                        gold_change = 2
            y_true.append(gold_change)
    
    for event in events:
        gold_change = 0
        for lst in pred:
            for t in lst:
                if t[0] in events:
                    if t[1] == 'more':
                        gold_change = 1
                    else:
                        gold_change = 2
            y_pred.append(gold_change)
    
assert len(y_true) == len(y_pred)

#%%
print(f1_score(y_pred, y_true, average='macro'))
print(accuracy_score(y_true, y_pred))

#conf_mtx = confusion_matrix(y_true, y_pred)
#ConfusionMatrixDisplay(conf_mtx, display_labels=['equally likely', 'more likely', 'less likely']).plot()
# %%