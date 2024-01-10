import os
import evaluate
import json
from tqdm import tqdm
import numpy as np
import json

import random
import re
import time
import openai



openai.api_key = os.getenv("OPENAI_API_KEY")
GPT_ENGINE="PLACE HOLDER"

def extract_keywords(results):
    """
    keyword_list: a list of entities
    keyword_indice_list: a list of indices
    """
    keyword_list = []
    keyword_indice_list = []
    cur_token = ""
    cur_indice = []
    for i, token in enumerate(results):
        if i != 0 and token["start"] == results[i - 1]["end"]:
            if '##' in token["word"]:
                cur_token += token["word"][2:]
            else:
                cur_token += token["word"]
            cur_indice.append(token["index"])
        else:
            if token["score"] > 0.8:
                if token["entity"][0] == "B":
                    if cur_token != "":
                        keyword_list.append(cur_token)
                        keyword_indice_list.append(cur_indice)
                    cur_token = token["word"]
                    cur_indice = [token["index"]]
                else:
                    if i == 0:
                        # error
                        continue
                    elif token["start"] == results[i - 1]["end"] + 1:
                        cur_token += " " + token["word"]
                        cur_indice.append(token["index"])
                    else:
                        # error
                        continue
    keyword_list.append(cur_token)
    keyword_indice_list.append(cur_indice)
    return keyword_list, keyword_indice_list

def generate(input_prompt, temperature=0, max_tokens=100,partition_id=None):
    result = None
    retry_interval_exp=1
    
    while True:
        try:
            if not partition_id:
                # Requests will be routed in round robin by default.
                partition_id = f"123"
            # response=gpt_model.complete(prompt=input_prompt,temperature=temperature,max_tokens=max_tokens)
            response = openai.Completion.create(
                # engine="gpt35",
                engine=GPT_ENGINE,
                # engine="text-davinci-003",
                prompt=input_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                logprobs=1,
                top_p=1,
                frequency_penalty=0.5,
                presence_penalty=0.0,
                stop=None,
                headers={"partition-id": partition_id},
            )
            result = [
                response["choices"][i]["text"].replace("\n", "").replace(" .", ".").strip()
                for i in range(len(input_prompt))
            ]
            return result
        except Exception as e:
            # NOTE: openai.error.RateLimitError: Requests to the
            # Deployments_Completion Operation under OpenAI API have
            # exceeded rate limit of your current OpenAI S0 pricing tier.
            # Please retry after 7 seconds. Please contact Azure support
            # service if you would like to further increase the default rate
            # limit.
            if isinstance(e, openai.error.APIConnectionError):
                # Expontial backoff
                time.sleep(max(4, 0.5 * (2**retry_interval_exp)))
                retry_interval_exp += 1
                print('apiconnection error')
            elif isinstance(e,openai.error.RateLimitError):
                # error = {"type": type(e).__name__, "message": str(e)}
                # print(error)
                # Expontial backoff
                time.sleep(max(4, 0.5 * (2 ** retry_interval_exp)))
                retry_interval_exp += 1
                print('RateLimitError')
            else:
                # NOTE: openai.error.InvalidRequestError: The response was
                # filtered due to the prompt triggering Azure OpenAI’s
                # content management policy.
                error = {"type": type(e).__name__, "message": str(e)}
                time.sleep(max(4, 0.5 * (2 ** retry_interval_exp)))
                retry_interval_exp += 1
                print(error)
                # return

def build_prompt_for_instruction(
        all_data,
        gpt_summ_entry='gpt_summ',
        fewshot_prompt=None
    ):

    all_prompts = []
    orig_summ=[]
    new_summ_mapping={}
    for i in range(len(all_data)):
        summ = all_data[i][gpt_summ_entry]
        doc=all_data[i]['article']
        orig_summ.append(summ)
        prompt = '''Document: %s \nSummary: %s \nThe summary may not cover the salient content, generate instructions to make the summary focus on salient content. 
            The instructions should be chosen from the following formats: 
            Delete content related to __. 
            Add content related to __. 
            No operation is needed.
            Only output the instructions without the corrected summaries, and make the instruction conservatively. 
            Instructions: ''' % (
            summ,
            doc,
        ) 
        if fewshot_prompt:
            new_prompt='\n\n'.join(fewshot_prompt)+'\n\n'+prompt
            i=len(fewshot_prompt)
            while len(new_prompt.split())>5000:
                i-=1
                new_prompt='\n\n'.join(fewshot_prompt[:i])+'\n\n'+prompt
            prompt = new_prompt
        new_summ_mapping[i]=len(all_prompts)
        all_prompts.append(prompt)
    return all_prompts,orig_summ,new_summ_mapping
def build_prompt_for_correction_with_instruction(
        all_data,
        instructions=None,
        gpt_summ_entry='gpt_summ'
    ):

    all_prompts = []
    orig_summ=[]
    new_summ_mapping={}
    for i in range(len(all_data)):
        summ = all_data[i][gpt_summ_entry]
        doc=all_data[i]['article']
        orig_summ.append(summ)
        if instructions[i]=='No operation is needed.':
            continue
        prompt = "Summary: %s \n Document: %s \n Rewrite the summary for the document, %s \n New summary: " % (
            summ,
            doc,
            instructions[i]
        ) 
        new_summ_mapping[i]=len(all_prompts)
        all_prompts.append(prompt)
    return all_prompts,orig_summ,new_summ_mapping

def build_prompt_for_correction(
        all_data,
        instructions=None,
        use_all_keywords=True,
        keyword_list_plus=None,
        keyword_list_minus=None,
        gpt_summ_entry='gpt_summ'
    ):
    if (keyword_list_minus is None) and (keyword_list_plus is None):
        summ = [all_data[i][gpt_summ_entry] for i in range(len(all_data))]
        keyword_list_plus,keyword_list_minus=build_keyword_list_from_output(instructions,orig_summaries= summ)
    all_prompts = []
    orig_summ=[]
    new_summ_mapping={}
    for i in range(len(all_data)):
        summ = all_data[i][gpt_summ_entry]
        doc=all_data[i]['article']
        orig_summ.append(summ)
        if len(keyword_list_plus[i])==0 and len(keyword_list_minus[i]) == 0:
            continue
        instructions=''
        if len(keyword_list_plus[i]) > 0:
            if use_all_keywords:
                instructions += " add content related to %s." % (" and ".join(keyword_list_plus[i]))
            else:
                instructions += " add content related to %s." % (random.choice(keyword_list_plus[i]))
        if len(keyword_list_minus[i]) > 0:
            if use_all_keywords:
                instructions += " delete content related to %s." % (" and ".join(keyword_list_minus[i]))
            else:
                instructions += " delete content related to %s." % (random.choice(keyword_list_minus[i]))
        prompt = "Summary: %s \n\nDocument: %s \n\nRewrite the summary for the document, %s \n\n New summary: " % (
            summ,
            doc,
            instructions
        ) 
        new_summ_mapping[i]=len(all_prompts)
        all_prompts.append(prompt)
    return all_prompts,orig_summ,new_summ_mapping

def build_keyword_list_from_output(instructions,orig_summaries=None):
    plus_list_all=[]
    minus_list_all=[]
    for i,instruction in enumerate(instructions):
        if orig_summaries:
            orig_summary=orig_summaries[i]
        plus_text=re.search(r'\<add\>(.*?)\<remove\>',instruction)
        if plus_text:
            plus_list=plus_text.group(1)
            plus_list=plus_list.split(',')
            plus_list=[kw.strip() for kw in plus_list if len(kw.strip())>0 and ('##' not in kw) ]
            # plus_list=[kw.strip() for kw in plus_list if orig_summary and kw not in orig_summary]
        else:
            plus_list=[]

        minus_text=re.search(r'\<remove\>(.*?)(\<remove\>|\<add\>)',instruction+'<remove>')
        if minus_text:
            minus_list=minus_text.group(1)
            minus_list=minus_list.split(',')
            minus_list=[kw.strip() for kw in minus_list if len(kw.strip())>0 and ('##' not in kw) ]
            # minus_list=[kw.strip() for kw in minus_list if orig_summary and kw in orig_summary]
        else:
            minus_list=[]
        plus_list_all.append(list(set(plus_list)))
        minus_list_all.append(list(set(minus_list)))
    return plus_list_all,minus_list_all

def compute_knowledge_f1(nlp, gt_summary_batch, predicted_summary_batch):
    all_gt_summ = [s for s in gt_summary_batch]
    all_gen_summ = [s for s in predicted_summary_batch]
    gt_summ_results = nlp(all_gt_summ)
    gen_summ_results = nlp(all_gen_summ)
    recall=[]
    precision=[]
    f1=[]
    for i_batch in range(len(gt_summ_results)):
        keyword_list_gt, _ = extract_keywords(gt_summ_results[i_batch])
        keyword_list_gen, _ = extract_keywords(gen_summ_results[i_batch])
        num_overlap=len(set(keyword_list_gt).intersection(set(keyword_list_gen)))
        r = num_overlap/(len(set(keyword_list_gt))+1e-9)
        p=num_overlap/(len(set(keyword_list_gen))+1e-9)
        f=2*r*p/(r+p+1e-9)
        if len(keyword_list_gen)==0 and len(keyword_list_gt)==0:
            r=1
            p=1
            f=1
        recall.append(r)
        precision.append(p)
        f1.append(f)
    return recall,precision,f1

def compute_knowledge_f1_all(nlp, all_gt_summ,all_predicted_summ):
    recall=[]
    precision=[]
    f1=[]
    for i in tqdm(range(0,len(all_gt_summ),32)):
        gt_summary_batch=all_gt_summ[i:i+32]
        predicted_summary_batch=all_predicted_summ[i:i+32]
        r,p,f=compute_knowledge_f1(nlp, gt_summary_batch, predicted_summary_batch)
        recall.extend(r)
        precision.extend(p)
        f1.extend(f)
    return recall,precision,f1

if __name__=='__main__':
    # Path to the generated instruction file
    instruction_file='place holder'
    # Where to store the generated summaries
    output_gptsumm_file='place holder'
    # Path to the dataset file
    dataset_file='./data/cnndm_chatgptsumm_with_kw_test.jsonl'
    # Path to the initial summary (iterative editing only), None if for the first iteration
    initial_summ_file=None

    # load test data
    all_data=[]
    with open(dataset_file,'r') as of:
        all_lines=of.readlines()
        for i,l in enumerate(all_lines):
            all_data.append(json.loads(l))
    
    if initial_summ_file:
        all_gpt_summ={}
        with open(initial_summ_file,'r') as of:
            all_lines=of.readlines()
            for l in all_lines:
                d = json.loads(l)
                all_gpt_summ[d['id']]=d['gpt_summ']
        for d in all_data:
            d['gpt_summ'] = all_gpt_summ[d['id']]

    rouge = evaluate.load('rouge')
    print(len(all_data))
    # load instructions
    all_kw_result={}
    with open(instruction_file,'r') as of:
        all_lines=of.readlines()
        for l in all_lines:
            d = json.loads(l)
            all_kw_result[d['id']]=d['gen_instruction']
    all_instructions=[]       
    for d in all_data:
        all_instructions.append(all_kw_result[d['id']])
    # build prompt
    all_prompts,orig_summ,new_summ_mapping = build_prompt_for_correction(all_data,
            all_instructions,
            use_all_keywords=True
        )
    # generate
    if len(all_prompts)>16:
        predicted=[]
        batches = [all_prompts[i:i+16] for i in range(0,len(all_prompts),16)]
        for b in tqdm(batches):
            predicted.extend(generate(b))
    else:
        predicted = generate(all_prompts)
    all_predictions=[predicted[new_summ_mapping[i]] if i in new_summ_mapping.keys() else orig_summ[i] for i in range(len(all_data))]
    predicted=all_predictions
    metric_results = rouge.compute(
        predictions=predicted, references=[d['highlights'] for d in all_data], use_stemmer=True
    )
    print(metric_results)
    # save results
    with open(output_gptsumm_file,'w') as of:
        for i,d in enumerate(all_data):
            json.dump({'id':d['id'],'gpt_summ':all_predictions[i]},of)
            of.write('\n')