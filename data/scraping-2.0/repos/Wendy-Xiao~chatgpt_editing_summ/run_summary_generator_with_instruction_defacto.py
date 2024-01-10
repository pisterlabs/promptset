import json
from datetime import datetime
import os
import random
import re
import time
import openai
from tqdm import tqdm
from nltk import sent_tokenize


OPENAI_API = os.getenv("OPENAI_API_KEY")
GPT_ENGINE=''
def generate(input_prompt, temperature=0, max_tokens=100,partition_id=None):
    result = None
    retry_interval_exp=1
    openai.api_key = OPENAI_API #os.getenv("OPENAI_API_KEY")
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


if __name__=='__main__':
    data_path = './data/DeFacto/'
    instruction_file=''
    output_gptsumm_file=''

    all_data=[]
    with open(data_path+'test.jsonl','r') as of:
        all_lines=of.readlines()
        for l in all_lines:
            all_data.append(json.loads(l))
    all_data_error=[d for d in all_data if len(d['feedback']['summary'])!=0]

    train_data=[]
    with open(data_path+'train.jsonl','r') as of:
        all_lines=of.readlines()
        for l in all_lines:
            train_data.append(json.loads(l))


    all_instructions={}
    with open(instruction_file,'r') as of:
        all_lines=of.readlines()
        for l in all_lines:
            d = json.loads(l)
            all_instructions[d['id']]=d['gen_instruction']
    all_predicted_instructions=[]       
    for d in all_data_error:
        all_predicted_instructions.append(all_instructions[d['doc_id']])
    all_prompts = []
    few_shot_example = ["Document: %s \nSummary: %s \nInstructions: %s \nEdit the summary only following the instructions and only output the corrected summary.\nNew summary: %s\n" % (
                d['article'],
                d['candidate'],
                d['feedback']['instruction'],
                d['feedback']['summary']
            ) for d in train_data[:14] if len(d['feedback']['summary'])!=0]

    for i,d in enumerate(all_data_error):
        prompt = "Document: %s \nSummary: %s \nInstructions: %s \nEdit the summary only following the instructions and only output the corrected summary.\nNew summary: " % (
            d['article'],
            d['candidate'],
            all_predicted_instructions[i]
        )
        prompt='\n\n'.join(few_shot_example)+'\n\n'+prompt
        all_prompts.append(prompt)

    if len(all_prompts)>16:
        predicted=[]
        batches = [all_prompts[i:i+16] for i in range(0,len(all_prompts),16)]
        for b in tqdm(batches):
            predicted.extend(generate(b,temperature=0, max_tokens=50))
    else:
        predicted = generate(all_prompts, temperature=0, max_tokens=50)
    all_predictions = [sent_tokenize(s)[0] for s in predicted]
    with open(output_gptsumm_file,'w') as of:
        for i,d in enumerate(all_data):
            json.dump({'id':d['id'],'gpt_summ':all_predictions[i]},of)
            of.write('\n')