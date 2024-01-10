import copy
import traceback
import time
import os
import traceback
import openai
import json

# put your api_key here.
api_key = ''
openai.api_key = api_key

# @timeout_decorator.timeout(30)
def call_gpt4(system_prompt, user_prompt):
    # for 10 tries
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", # gpt-3.5-turbo
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.01
    )
    result = completion["choices"][0]["message"]["content"]
    return result

def check_pre_isin(pre, pre_result):
    if pre_result is None:
        return False

    else:
        for elem in pre_result['pre']:
            if pre == elem['pred_input']:
                return True
        return False

def calculate_precision(pred_list, gold_list, cur_result):
    system_prompt = "You are excellent video reasoner. You are given caption and caption list. Although you can't see the video, imagine the video scene as much as possible based on caption list. You should determine whether the caption in appears in the scene of the video, according to your judgment. Even if there is no sentence in the caption list that matches exactly, if you can infer that it could reasonably appear in the video scene, you should consider it as appearing. You should focus more on the semantic similarity between the sentences when making judgments. Please answer yes or no."
    if cur_result is not None:
        toreturn = copy.deepcopy(cur_result['pre'])
    else:
        toreturn = []

    for ii, pre in enumerate(pred_list):
        if not check_pre_isin(pre, cur_result):    
            user_prompt = f"Answer captions: {gold_list}\nPredicted caption: {pre}\nAnswer:"
            print(f'in calculate_precision calling {ii}/{len(pred_list)} len user_prompt {len(user_prompt)} len system_prompt {len(system_prompt)}')
            result = call_gpt4(system_prompt, user_prompt)
            toreturn.append({'result':result, 'pred_input':pre, 'gold_list':gold_list})    
            time.sleep(2)
        # vkey, pre/recall, system_prompt, user_prompt
        

    return toreturn

def check_recall_isin(ans, recall_result):  
    if recall_result is None:
        return False
    else:
        for elem in recall_result['recall']:
            if ans == elem['gold_input']:
                return True
        return False
    


def calculate_recall(pred_list, gold_list, cur_result):
    system_prompt = "You are excellent video reasoner. You are given caption and caption list. Although you can't see the video, imagine the video scene as much as possible based on caption list. You should determine whether the caption in appears in the scene of the video, according to your judgment. Even if there is no sentence in the caption list that matches exactly, if you can infer that it could reasonably appear in the video scene, you should consider it as appearing. You should focus more on the semantic similarity between the sentences when making judgments. Please answer yes or no."
    if cur_result is not None:
        toreturn = copy.deepcopy(cur_result['recall'])
    else:
        toreturn = []

    for ii, ans in enumerate(gold_list):
        if not check_recall_isin(ans, cur_result):
            user_prompt = f"Predicted captions: {pred_list}\nAnswer caption: {ans}\nAnswer:"
            print(f' in calculate_recall calling {ii}/{len(gold_list)} len user_prompt {len(user_prompt)} len system_prompt {len(system_prompt)} len predlist {len(json.dumps(pred_list))} ')
            result = call_gpt4(system_prompt, user_prompt)
            toreturn.append({'result':result, 'gold_input':ans, 'pred_list':pred_list})
            time.sleep(2)
    return toreturn


def interrupt_safe_save(save_fp, results):
    try:
        with open(save_fp, 'w') as fp:
            json.dump(results, fp, indent=4)
        
    except Exception as E:
        if E is KeyboardInterrupt:
            print('We are saving it! please wait Never interrupt again!')
            with open(save_fp, 'w') as fp:
                json.dump(results, fp, indent=4)
            print('saving done!')
            exit()
            
if __name__ == "__main__":
    # readme input for /home/ubuntu/kael/evaluation/video_chat_v2_ours_improved.json
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_fp', type=str)
    parser.add_argument('--gold_fp', type=str)
    parser.add_argument('--save_fp', type=str)
    parser.add_argument('--gpttype', type=str, default='gpt-4')

    args = parser.parse_args()
    pred_fp = args.pred_fp
    gold_fp = args.gold_fp #'/home/ubuntu/ray/gpt-based-evaluation/gold_caption_atomized.json'
    save_fp = args.save_fp
    
    with open(pred_fp, 'r') as fp:
        pred = json.load(fp)
    
    with open(gold_fp, 'r') as fp:
        gold = json.load(fp)
    
    if os.path.exists(save_fp):
        with open(save_fp, "r") as f:
            results = json.load(f)
    else:
        results = {}

    pkeys = set(pred.keys())
    gkeys = set(gold.keys())
    common_keys = pkeys.intersection(gkeys)

    num_p_error = 0
    num_g_error = 0

    for vii, vkey in enumerate(common_keys):
        # orig_vkey = vkey
        # vkey = vkey.replace('.mp4', '').replace('_mp4','')
        # condition_met = orig_vkey in pred and vkey in gold
           
        _pred = pred[vkey]['pred_atomic']
        _gold = gold[vkey]['pred_atomic']
        # _pred = eval(_pred)  
        # _gold = eval(_gold)
        try:
            _pred = json.loads(_pred)
        except:
            traceback.print_exc()
            _pred = []
            num_p_error += 1

        try:
            _gold = json.loads(_gold)
        except:
            traceback.print_exc()
            _gold = []
            num_g_error += 1

        print(f'num_p_error {num_p_error} num_g_error {num_g_error}')
        if vkey in results:
            cur_result = results[vkey]
        #     print('skipping because already processed', vkey)
        #     continue
        else:
            cur_result = None
        try:
            _precision_result = calculate_precision(_pred, _gold, cur_result)
            _recall_result = calculate_recall(_pred, _gold, cur_result)
            results[vkey] = {'pre':_precision_result, 'recall':_recall_result}
            
        except Exception as E:
            if E is KeyboardInterrupt:
                raise E
            traceback.print_exc()
            time.sleep(60)            
        
        interrupt_safe_save(save_fp, results)
