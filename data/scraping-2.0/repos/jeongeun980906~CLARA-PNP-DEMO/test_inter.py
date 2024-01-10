import openai
import random
import spacy
import scipy
import copy
import numpy as np
import json
from llm.affordance import affordance_scoring, affordance_score2
from llm.lnct_score import lm_planner_unct # GPT3
# from llm.llama import llama_unct
from llm.chat import lm_planner_unct_chat
from eval.run import run
import argparse
import numpy as np
import random
import cv2
from env.success import *
THRES = {
    'gpt_1': 1.43, 
    'gpt_7': 0.5, 
    'gpt_2': 2.67,
    'chat_2':2.10,'chat_7': 0.5, 
}
def set_openai_api_key_from_txt(key_path='./key.txt',VERBOSE=True):
    """
        Set OpenAI API Key from a txt file
    """
    with open(key_path, 'r') as f: 
        OPENAI_API_KEY = f.read()
    openai.api_key = OPENAI_API_KEY
    if VERBOSE:
        print ("OpenAI API Key Ready from [%s]."%(key_path))
    

def only_few_shot_question(res, lm_planner:lm_planner_unct):
    for new_name, task in res.items():
        goal = task['name']
        found_objects = task['found_objects']
        lm_planner.reset()
        lm_planner.objects = copy.deepcopy(found_objects)
        lm_planner.set_goal(goal)
        lm_planner.set_prompt()
        steps = lm_planner.infer_wo_unct(found_objects,'ood')
        print(steps)
        done = True
        if len(steps)>0:
            if "question:" in steps[-1]:
                res[new_name]['answer'] = [None]
                done = False
        res[new_name]['steps'] = steps
        res[new_name]['done'] = done
    return res

def only_few_shot_answer(res, lm_planner:lm_planner_unct):
    for new_name, task in res.items():
        goal = task['name']
        found_objects = task['found_objects']
        done = task['done']
        lm_planner.reset()
        if not done:
            previous_step = task['steps']
            lm_planner.objects = copy.deepcopy(found_objects)
            lm_planner.set_goal(goal)
            lm_planner.set_prompt()
            for line in previous_step:
                lm_planner.new_lines += "\n"+line
            lm_planner.answer(task['answer'][0])
            new_steps = lm_planner.infer_wo_unct(found_objects,'ood',stop=False)
            print(new_steps)
            res[new_name]['steps'] = previous_step + new_steps
    return res

def first_trial(res, lm_planner:lm_planner_unct, thres:float):
    # temp = 0
    for new_name, task in res.items():
        # temp += 1
        # if temp > 10:
        #     break
        goal = task['name']
        found_objects = task['found_objects']
        uncts = task['uncertainties']
        steps = task['steps']
        lm_planner.reset()
        lm_planner.objects = copy.deepcopy(found_objects)
        lm_planner.set_goal(goal)
        lm_planner.set_prompt()
        done = True
        new_steps = []
        new_uncts = []
        for idx, (unct, step) in enumerate(zip(uncts, steps)):
            unct = unct['total']
            new_steps.append(step)
            new_uncts.append(unct)
            lm_planner.append(None, None, step)
            if unct > thres:
                reason, ques = lm_planner.question_generation()
                done = False
                break
            
        res[new_name]['done'] = done
        res[new_name]['new_step'] = new_steps
        res[new_name]['new_uncts'] = new_uncts
        if not done:
            res[new_name]['reason'] = [reason]
            res[new_name]['question'] = [ques]
            res[new_name]['answer'] = [None]
            res[new_name]['idx'] = [idx]
    return res

def run(res, lm_planner:lm_planner_unct, thres:float, last_trial: bool = False):
    max_tasks = 5
    # temp = 0
    for new_name, task in res.items():
        # temp += 1
        # if temp > 10:
        #     break
        goal = task['name']
        found_objects = task['found_objects']
        gt_objects = task['gt_objects']+found_objects
        uncts = task['new_uncts']
        steps = task['new_step']
        lm_planner.reset()
        lm_planner.objects = copy.deepcopy(found_objects)
        lm_planner.set_goal(goal)
        lm_planner.set_prompt()
        done = task['done']

        if not done:
            answers = task['answer']
            assert None not in answers
            last = 0
            idxs = task['idx']
            reasons = task['reason']
            questions = task['question']
            for idx, reason, question,answer in zip(idxs,reasons, questions,answers):
                idx += 1
                lm_planner.append(None, None, steps[last])
                for step in steps[last+1:idx]:
                    lm_planner.append(None, None, step)
                lm_planner.append_reason_and_question(reason, question)
                lm_planner.answer(answer)
                last = idx
            num_tasks = last
            while not done:
                num_tasks += 1
                if num_tasks > max_tasks:
                    break
                tasks, scores , unct = lm_planner.plan_with_unct()
                if tasks != None:
                    scores = np.asarray(scores)
                    idxs= np.argsort(scores)
                    for idx in idxs[::-1]:
                        try:
                            aff = affordance_score2(tasks[idx], gt_objects)
                        except:
                            aff = 0
                        if aff > 0:
                            break
                    if aff == 2: 
                        done=True 
                        break
                if unct['total'] > thres and not last_trial:
                    reason, ques = lm_planner.question_generation()
                    reasons.append(reason)
                    questions.append(ques)
                    answers.append(None)
                    steps.append(tasks[idx])
                    uncts.append(unct['total'])
                    break
                steps.append(tasks[idx])
                uncts.append(unct['total'])
                lm_planner.append(None, None, tasks[idx])
            print(steps, done)
            res[new_name]['done'] = done
            res[new_name]['new_step'] = steps
            res[new_name]['new_uncts'] = uncts
            res[new_name]['reason'] = reasons
            res[new_name]['question'] = questions
            res[new_name]['answer'] = answers
            if not done:
                res[new_name]['idx'].append(num_tasks-1)
    return res
        
def main(args):
    set_openai_api_key_from_txt()
    random.seed(0)
    np.random.seed(0)

    unct_type = int(args.method)
    lm_type = args.lm
    if lm_type == 'gpt':
        lm_planner = lm_planner_unct(type=int(unct_type),example=args.few_shot)
    elif lm_type == 'chat':
        lm_planner = lm_planner_unct_chat(example=args.few_shot)    
    else:
        raise NotImplementedError
    try:
        thres = THRES['{}_{}'.format(lm_type, unct_type)]
    except:
        thres = None
    if unct_type == 1:
        unct_type = 'ent'
    elif unct_type == 7:
        unct_type = 'lu'
    elif unct_type == 2:
        unct_type = 'var'
    last_trial = args.last

    if args.few_shot:
        if args.first:
            with open('./res/pick_and_place_unct_vild.json','r') as f:
                res = json.load(f)
            res = only_few_shot_question(res, lm_planner)
        else:
            with open('./res/pick_and_place_unct_{}_fewshot_inter.json'.format(lm_type),'r') as f:
                res = json.load(f)
            res = only_few_shot_answer(res, lm_planner)

        with open('./res/pick_and_place_unct_{}_fewshot_inter.json'.format(lm_type),'w') as f:
            json.dump(res, f, indent=4)
        return

    elif args.first:
        with open('./res/pick_and_place_unct_{}_{}.json'.format(lm_type, unct_type),'r') as f:
            res = json.load(f)
        res = first_trial(res, lm_planner, thres)
    else:
        with open('./res/pick_and_place_unct_{}_{}_inter.json'.format(lm_type, unct_type),'r') as f:
            res = json.load(f)
        res = run(res, lm_planner, thres,last_trial)

    with open('./res/pick_and_place_unct_{}_{}_inter.json'.format(lm_type, unct_type),'w') as f:
        json.dump(res, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='1')
    parser.add_argument('--lm', type=str, default='gpt')
    parser.add_argument('--first',default=False, action='store_true')
    parser.add_argument('--last',default=False, action='store_true')
    parser.add_argument('--few_shot',default=False, action='store_true')
    args = parser.parse_args()
    main(args)