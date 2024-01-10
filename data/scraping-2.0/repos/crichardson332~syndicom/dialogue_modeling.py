import json
import pdb
import util
import os
from tqdm import tqdm
from pprint import pprint
import numpy as np
from pathlib import Path
import openai
import time


def generate_dialogue_responses(split):
    # openai api key
    api_key_file = Path('/Users/crichardson8/.gpt/api_key')
    with api_key_file.open() as f:
        key = f.readline()
        openai.api_key = key.replace('\n','')

    infile = f'output/dataset/{split}.jsonl'
    with open(infile, 'r') as json_file:
        json_list = list(json_file)

    outfile = f'output/dialogue_modeling/{split}.jsonl'

    #Generate function to get last line of infile
    with open(outfile, 'r') as f:
        lines = f.readlines()
        if len(lines) < 1:
            start_idx = 0
        else:
            start_idx = json.loads(lines[-1])['id'] + 1

    gpt_preamble = 'You will be given a dialogue context and your task is to continue the dialogue by generating a single response. The dialogue is between two people, and the speaker alternates every turn. Generate only one dialogue turn.\nContext:\n'


    gpt_ft_preamble = 'You will be given a dialogue context and your task is to continue the dialogue by generating a single response. The dialogue is between two people, and the speaker alternates every turn. Generate only one dialogue turn. Generate a dialogue turn such that an evaluator would give positive feedback and say the dialogue sounds very natural and normal.\nContext:\n'

    print(f'Generating responses for {split} split...')
    for idx,json_str in enumerate(tqdm(json_list)):
        if idx < start_idx:
            continue
        datum = json.loads(json_str)
        prompt = gpt_preamble + '\n'.join(datum['context']) + '\nResponse:\n'

        # ping GPT api
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=256,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=['Context:', 'Response:']
        )
        datum['response']['gpt'] = response['choices'][0]['text']

        ft_prompt = gpt_ft_preamble + '\n'.join(datum['context']) + '\nResponse:\n'
        # use finetuned model as well
        ft_model = "davinci:ft-georgia-institute-of-technology-2023-04-04-01-25-47"
        response_finetuned = openai.Completion.create(
            model=ft_model,
            prompt=prompt,
            temperature=0.7,
            max_tokens=50,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=['Context:', 'Response:', 'Explanation:', '\n']
        )
        datum['response']['gpt_finetuned'] = response_finetuned['choices'][0]['text']

        # write the new data to file
        with open(outfile, 'a') as f:
            json.dump(datum, f)
            f.write('\n')
    print(f'Generating responses for {split} split...Done')


def generate_flanT5(split):
    # this requires being on GPU servers
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

    # setup IO
    infile = f'output/dataset/{split}.jsonl'
    with open(infile, 'r') as json_file:
        json_list = list(json_file)

    outfile = f'output/dialogue_modeling/flanT5-{split}.jsonl'
    #Generate function to get last line of infile
    with open(outfile, 'a+') as f:
        lines = f.readlines()
        if len(lines) < 1:
            start_idx = 0
        else:
            start_idx = json.loads(lines[-1])['id'] + 1

    # preamble = 'Give feedback for the following AT-generated dialogue: \nDialogue:\n'
    preamble = 'Generate a single dialogue turn in response to the following dialogue context, such that critical feedback to the resulting synthetic dialogue will be positive and described the dialogue as natural. \nContext:\n'
    print(f'Generating responses for {split} split...')
    for idx,json_str in enumerate(tqdm(json_list)):
        if idx < start_idx:
            continue
        datum = json.loads(json_str)
        prompt = preamble + '\n'.join(datum['context'])

        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs)
        print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        pdb.set_trace()

def generate_corrections(split):

    openai.api_key = util.get_openai_key()

    infile = f'output/feedback/{split}.jsonl'
    with open(infile, 'r') as json_file:
        json_list = list(json_file)

    outfile = f'output/dialogue_modeling/corrections/{split}.jsonl'
    # with open(outfile, 'w') as f:
    #     pass
    start_idx = util.get_jsonl_current_idx(outfile)

    print(f'Generating responses for {split} split...')
    for idx,json_str in enumerate(tqdm(json_list)):
        if idx < start_idx:
            continue
        datum = json.loads(json_str)

        # rephrase the invalid response with chatgpt
        old_invalid = [resp['text'] for resp in datum['response'] if resp['source'] == 'invalid'][0]
        while True:
            try:
                resp_chatgpt = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Rephrase the following sentence. You can reorder phrases if you need, just keep the overall semantic meaning the same, and try to keep the length roughly the same as well:\n"},
                        {"role": "user", "content": old_invalid},
                    ]
                )
            except openai.error.RateLimitError:
                print('Rate limit exceeded, trying again...')
                continue
            break
        new_invalid = resp_chatgpt['choices'][0]['message']['content']

        # use the rephrased invalid
        for resp in datum['response']:
            if resp['source'] == 'invalid':
                resp['text'] = new_invalid
        prompts = util.get_gpt_correction_prompt(datum)
        instructions = 'Given the following dialogue context, baseline response, and feedback, generate an improved response over the baseline using the given feedback.\n'

        for prompt in prompts:
            # ping GPT api
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=(instructions + prompt + '\nImproved Response:\n'),
                temperature=0.7,
                max_tokens=256,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=['###', 'Baseline:', 'Feedback:', '\n']
            )
            datum['response'].append({
                'text': response['choices'][0]['text'],
                'source': 'gpt-base',
            })

            # use finetuned model as well
            model_ft = "davinci:ft-georgia-institute-of-technology-2023-04-06-19-34-21"
            response_ft = openai.Completion.create(
                model=model_ft,
                prompt=prompt,
                temperature=0.7,
                max_tokens=50,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=['###', 'Baseline:', 'Feedback:', '\n']
            )
            datum['response'].append({
                'text': response_ft['choices'][0]['text'],
                'source': 'gpt-ft-davinci',
            })

        # write the new data to file
        with open(outfile, 'a') as f:
            json.dump(datum, f)
            f.write('\n')
    print(f'Generating responses for {split} split...Done')

def gen_feedback(split):
    openai.api_key = util.get_openai_key()

    infile = f'output/feedback/{split}.jsonl'
    with open(infile, 'r') as json_file:
        json_list = list(json_file)

    outfile = f'output/dialogue_modeling/feedback/{split}.jsonl'
    start_idx = util.get_jsonl_current_idx(outfile)

    print(f'Generating responses for {split} split...')
    for idx,json_str in enumerate(tqdm(json_list)):
        if idx < start_idx:
            continue
        datum = json.loads(json_str)
        datum['feedback'] = []

        prompt = util.get_gpt_feedback_prompt(datum)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are shown a synthetic dialogue written by an AI. The dialogue is intended to sound like a natural text message conversation between two people. The AI is imperfect and makes mistakes. You are asked to provide feedback to the AI to improve its dialogue generation. You are given a few dialogue turns, followed by a Baseline Response. Please give 1-2 sentences of feedback for the baseline response, and please be specific!\n"},
                {"role": "user", "content": prompt + '\nFeedback:\n'},
            ]
        )
        datum['feedback'].append({
            'text': response['choices'][0]['message']['content'],
            'source': 'gpt-3.5-turbo',
        })

        # use finetuned model as well
        model_ft = "davinci:ft-georgia-institute-of-technology:feedback-2023-04-07-22-25-47"
        response_ft = openai.Completion.create(
            model=model_ft,
            prompt=prompt,
            temperature=0.7,
            max_tokens=50,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=['###', 'Baseline:', 'Feedback:', '\n']
        )
        datum['feedback'].append({
            'text': response_ft['choices'][0]['text'],
            'source': 'gpt-ft-davinci',
        })

        # write the new data to file
        with open(outfile, 'a') as f:
            json.dump(datum, f)
            f.write('\n')
    print(f'Generating responses for {split} split...Done')

def gen_responses_with_self_correction(split):
    openai.api_key = util.get_openai_key()

    infile = f'output/dataset/{split}.jsonl'
    with open(infile, 'r') as json_file:
        json_list = list(json_file)

    outfile = f'output/dialogue_modeling/self_correction/{split}.jsonl'
    start_idx = util.get_jsonl_current_idx(outfile)

    print(f'Generating responses for {split} split...')
    for idx,json_str in enumerate(tqdm(json_list)):
        if idx < start_idx:
            continue
        datum = json.loads(json_str)

        # baseline
        prompt = util.get_response_prompt(datum)
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You will be given a dialogue context and your task is to continue the dialogue by generating a single response. The dialogue is between two people, and the speaker alternates every turn. Generate only one dialogue turn.\n"},
                        {"role": "user", "content": prompt + '\nFeedback:\n'},
                    ]
                )
            except openai.error.RateLimitError:
                print('Rate limit exceeded, trying again...')
                continue
            break
        datum['response'].append({
            'text': response['choices'][0]['message']['content'],
            'source': 'gpt-3.5-turbo',
        })

        # self-correction steps: 
        # 1. generate generic response with stock GPT
        # 2. generate feedback with feedback model
        # 3. fix response with self-correction model

        # STEP 1
        gpt_preamble = 'You will be given a dialogue context and your task is to continue the dialogue by generating a single response. The dialogue is between two people, and the speaker alternates every turn. Generate only one dialogue turn.\nContext:\n'
        prompt = gpt_preamble + '\n'.join(datum['context']) + '\nResponse:\n'
        response = openai.Completion.create(
            engine="text-ada-001",
            prompt=prompt,
            temperature=0.7,
            max_tokens=256,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=['Context:', 'Response:']
        )
        baseline_response = response['choices'][0]['text']
        datum['response'].append({
            'text': response['choices'][0]['text'],
            'source': 'text-ada-001',
        })

        # STEP 2
        # prompt = util.get_gpt_feedback_prompt(datum, response['choices'][0]['text'])
        prompt = util.get_gpt_feedback_prompt(datum, response['choices'][0]['text'])
        model_fb = "davinci:ft-georgia-institute-of-technology:feedback-2023-04-07-22-25-47"
        response_fb = openai.Completion.create(
            model=model_fb,
            prompt=prompt,
            temperature=0.7,
            max_tokens=50,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=['###', 'Baseline:', 'Feedback:', '\n']
        )
        datum['feedback'] = [{
            'text': response_fb['choices'][0]['text'],
            'source': 'gpt-ft-feedback',
        }]

        # STEP 3
        prompt = '\n'.join(datum['context'])
        prompt += '\nBaseline Response:\n' + response['choices'][0]['text'] 
        prompt += '\nFeedback:\n' + response_fb['choices'][0]['text']
        prompt += '\n\n###\n\n'

        # FIXME get the new model here
        model_cr = "davinci:ft-georgia-institute-of-technology-2023-04-06-19-34-21"
        response_cr = openai.Completion.create(
            model=model_cr,
            prompt=prompt,
            temperature=0.7,
            max_tokens=50,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=['###', 'Baseline:', 'Feedback:', '\n']
        )
        datum['response'].append({
            'text': response_cr['choices'][0]['text'],
            'source': 'gpt-ft-correction',
        })

        # write the new data to file
        with open(outfile, 'a+') as f:
            json.dump(datum, f)
            f.write('\n')
    print(f'Generating responses for {split} split...Done')

def gen_feedback_and_correction(split):
    openai.api_key = util.get_openai_key()

    infile = f'output/dataset/{split}.jsonl'
    with open(infile, 'r') as json_file:
        json_list = list(json_file)

    outfile = f'output/dialogue_modeling/fb_correction/{split}.jsonl'
    start_idx = util.get_jsonl_current_idx(outfile)

    print(f'Generating responses for {split} split...')
    for idx,json_str in enumerate(tqdm(json_list)):
        if idx < start_idx:
            continue
        datum = json.loads(json_str)
        # rephrase the invalid response with chatgpt
        old_invalid = [resp['text'] for resp in datum['response'] if resp['source'] == 'invalid'][0]
        while True:
            try:
                resp_chatgpt = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Rephrase the following sentence. You can reorder phrases if you need, just keep the overall semantic meaning the same, and try to keep the length roughly the same as well:\n"},
                        {"role": "user", "content": old_invalid},
                    ]
                )
            except openai.error.RateLimitError:
                print('Rate limit exceeded, trying again...')
                continue
            break
        new_invalid = resp_chatgpt['choices'][0]['message']['content']

        # use the rephrased invalid
        for resp in datum['response']:
            if resp['source'] == 'invalid':
                resp['text'] = new_invalid
        
        # baseline
        prompt = util.get_response_prompt(datum)
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You will be given a dialogue context and a baseline response. Your job is to improve that baseline response. Do so by first generating feedback for that response, as if it was written by an AI and you are critiquing it, and then produce the improved response. Always write the improved response last and prefix it with 'Improved Response:'\n"},
                        {"role": "user", "content": prompt + '\nBaseline Response:\n' + new_invalid + '\n'},
                    ]
                )
            except openai.error.RateLimitError:
                print('Rate limit exceeded, trying again...')
                continue
            break
        chatgpt_responses = response['choices'][0]['message']['content'].split("Improved Response:")
        chatgpt_fb = chatgpt_responses[0]
        chatgpt_ir = chatgpt_responses[1]
        datum['response'].append({
            'text': chatgpt_ir,
            'source': 'gpt-3.5-turbo',
        })
        datum['feedback'] = [{
            'text': chatgpt_fb,
            'source': 'gpt-3.5-turbo',
        }]

        # baseline 2: direct improvement
        model_direct = "davinci:ft-georgia-institute-of-technology:negation-2023-05-20-19-31-47"
        response_direct = openai.Completion.create(
            model=model_direct,
            prompt=new_invalid,
            temperature=0.7,
            max_tokens=50,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=['###', 'Baseline:', 'Feedback:', '\n']
        )
        datum['response'].append({
            'text': response_direct['choices'][0]['text'],
            'source': 'gpt-ft-direct',
        })

        # self-correction steps: 
        # 1. rephrase invalid response
        # 2. generate feedback with feedback model
        # 3. fix response with self-correction model

        # STEP 2
        # prompt = util.get_gpt_feedback_prompt(datum, response['choices'][0]['text'])
        prompt = util.get_gpt_feedback_prompt(datum, new_invalid)
        model_fb = "davinci:ft-georgia-institute-of-technology:feedback-2023-04-07-22-25-47"
        response_fb = openai.Completion.create(
            model=model_fb,
            prompt=prompt,
            temperature=0.7,
            max_tokens=50,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=['###', 'Baseline:', 'Feedback:', '\n']
        )
        datum['feedback'].append({
            'text': response_fb['choices'][0]['text'],
            'source': 'gpt-ft-feedback',
        })

        # STEP 4
        prompt = '\n'.join(datum['context'])
        # prompt += '\nBaseline Response:\n' + response['choices'][0]['text'] 
        prompt += '\nBaseline Response:\n' + new_invalid
        prompt += '\nFeedback:\n' + response_fb['choices'][0]['text']
        prompt += '\n\n###\n\n'

        # FIXME get the new model here
        model_cr = "davinci:ft-georgia-institute-of-technology-2023-04-06-19-34-21"
        response_cr = openai.Completion.create(
            model=model_cr,
            prompt=prompt,
            temperature=0.7,
            max_tokens=50,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=['###', 'Baseline:', 'Feedback:', '\n']
        )
        datum['response'].append({
            'text': response_cr['choices'][0]['text'],
            'source': 'gpt-ft-correction',
        })

        # write the new data to file
        with open(outfile, 'a+') as f:
            json.dump(datum, f)
            f.write('\n')
    print(f'Generating responses for {split} split...Done')


if __name__ == '__main__':
    # splits = ['train','val','test']
    splits = ['test']
    # splits = ['dev']
    # splits = ['train']
    for split in splits:
        # generate_dialogue_responses(split)
        # generate_flanT5(split)
        # generate_corrections(split)
        # gen_feedback(split)
        # gen_responses_with_self_correction(split)
        gen_feedback_and_correction(split)