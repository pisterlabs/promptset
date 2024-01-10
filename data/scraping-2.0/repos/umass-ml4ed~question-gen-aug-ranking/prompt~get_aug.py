'''
python get_aug.py -NK 6 -MN code-davinci-002 -D -MT 32 -T 0.8 -P 0.9 -N 2
'''
import os
import argparse
import time
from tqdm import tqdm
import json
import pandas as pd
import numpy as np
import openai
from openai.error import RateLimitError, Timeout, APIError, ServiceUnavailableError


# set api keys
def set_api_keys():
    # TODO: Replace with your API keys
    os.environ['OPENAI_API_KEY_1'] = 'Replace with key-1'
    os.environ['OPENAI_API_KEY_2'] = 'Replace with key-2'
    os.environ['OPENAI_API_KEY_3'] = 'Replace with key-3'
    os.environ['OPENAI_API_KEY_4'] = 'Replace with key-4'
    os.environ['OPENAI_API_KEY_5'] = 'Replace with key-5'
    os.environ['OPENAI_API_KEY_6'] = 'Replace with key-6'


# Collect api keys
def get_api_keys(num_keys):
    api_keys = []
    for i in range(num_keys):
        api_key = os.getenv("OPENAI_API_KEY_{:d}".format(i+1))
        if api_key:
            api_keys.append(api_key)
    return api_keys


# construct input prompt 
def construct_input_prompt(context, answer, attribute, question, ex_im=None):
    prompt = '<Context> {:s}\n'.format(context)
    prompt +=  '<Answer> {:s}\n'.format(answer)
    prompt +=  '<Attribute> {:s}\n'.format(attribute)
    if ex_im is not None:
        prompt +=  '<Ex_or_Im> {:s}\n'.format(ex_im)
    prompt +=  '<Question> {:s}\n'.format(question)
    prompt +=  'Another question with the same answer is: '
    return prompt


###############################################
# NOTE: Response generation
delay_time = 5
decay_rate = 0.8
key_index = 0


def run_model(prompt, args, api_keys):
    global delay_time 
    global key_index 

    # sleep to avoid rate limit error
    time.sleep(delay_time)

    # alternate keys to avoid rate limit for codex
    print('Using Key Index {:d} and Key {:s}'.format(key_index, api_keys[key_index]))
    openai.api_key = api_keys[key_index]
    key_index = (key_index + 1) % len(api_keys)

    try:
        response = openai.Completion.create(
            model = args.model_name,
            prompt = prompt,
            max_tokens = args.max_tokens, 
            temperature = args.temperature,
            top_p = args.top_p,
            n = args.num_samples, 
#            best_of = 1, 
            stop = [args.stop]
        )

        delay_time *= decay_rate
    except (RateLimitError, Timeout, APIError, ServiceUnavailableError) as exc:
        print(exc)
        delay_time *= 2
        return run_model(prompt, args, api_keys)
    
    return response


def add_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-SD', '--selective_augmentation', action=argparse.BooleanOptionalAction, help='Augment all attribute except action and causal')
    parser.add_argument('-EXIM', '--exim', action=argparse.BooleanOptionalAction, help='Augment by considering explicit and implicit tags')
    parser.add_argument("-NK", "--num_keys", type=int, default=6, help="Number of API keys to use")
    parser.add_argument("-FN", "--fold_number", type=int, default=1, help="Fold Decoding")
    parser.add_argument("-MN", "--model_name", type=str, default="code-davinci-002", help="GPT-3 off-the-shelf or finetuned model name")
    parser.add_argument('-D', '--debug', action='store_true', help='Debug mode for augmenting on a small subset of 5 samples')
    # Codex generation parameters
    parser.add_argument("-MT", "--max_tokens", type=int, default=32, help="Maximum number of tokens to generate")
    parser.add_argument("-T", "--temperature", type=float, default=0.8, help="Temperature for sampling")
    parser.add_argument("-P", "--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("-N", "--num_samples", type=int, default=2, help="Number of samples to generate")
    parser.add_argument("-BO", "--best_of", type=int, default=1, help="Number of samples to take the best n, best_of must be >= n")
    parser.add_argument('-S', "--stop", type=str, default='<END>', help="Stop sequence")

    params = parser.parse_args()
    return params


def main():
    args = add_params()
    # set api keys
    set_api_keys()
    # get api keys
    api_keys = get_api_keys(args.num_keys)
    if args.fold_number % 2 != 0:
        api_keys = api_keys[:len(api_keys)//2]
    else:
        api_keys = api_keys[len(api_keys)//2:]

    # NOTE: read test data
    print('Loading Fold {:d}'.format(args.fold_number))
    test_file_path = '../data/FairytaleQA_story_folds/{:d}/test.json'.format(args.fold_number)
    test_data = []
    with open(test_file_path, 'r') as infile:
        for line in infile:
            test_data.append(json.loads(line))
    if args.debug:
        test_df = pd.DataFrame(test_data)[:10]
    else:
        test_df = pd.DataFrame(test_data)
    
    # attribute check
    allow_attr = list(test_df['attribute'].unique())
    if args.selective_augmentation:
        try:
            allow_attr.remove('action')
        except ValueError:
            pass
        try:
            allow_attr.remove('causal relationship')
        except ValueError:
            pass

    # NOTE: Read QG prompt (attribute-wise)
    ini_prompts = {}    
    if args.selective_augmentation:
        prompt_dir = 'prompt_dir/aq_prompt'
        for attr in allow_attr:
            with open(os.path.join(prompt_dir, '{:s}.txt'.format(attr)), 'r') as infile:
                attr_prompt = infile.read()
            ini_prompts[attr] = attr_prompt
    elif args.exim:
        prompt_dir = 'prompt_dir/aq_prompt_exim'
        exim = ['explicit', 'implicit']
        for attr in allow_attr:
            for choice in exim:
                with open(os.path.join(prompt_dir, '{:s}_{:s}.txt'.format(attr, choice)), 'r') as infile:
                    attr_exim_prompt = infile.read()
                ini_prompts['{:s}_{:s}'.format(attr, choice)] = attr_exim_prompt
    else:
        if not args.selective_augmentation and not args.exim:
            with open('prompt_dir/aq_prompt.txt', 'r') as infile:
                main_prompt = infile.read()
            for attr in allow_attr:
                ini_prompts[attr] = main_prompt

    # get input prompts
    ip_prompts = []
    for i in range(len(test_df)):
        if test_df.loc[i, 'attribute'] in allow_attr:
            if args.exim:
                prompt = construct_input_prompt(test_df.loc[i, 'content'], test_df.loc[i, 'answer'], test_df.loc[i, 'attribute'], test_df.loc[i, 'question'], test_df.loc[i, 'ex_or_im'])
                ip_prompts.append(ini_prompts['{:s}_{:s}'.format(test_df.loc[i, 'attribute'], test_df.loc[i, 'ex_or_im'])] + prompt)
            else:
                prompt = construct_input_prompt(test_df.loc[i, 'content'], test_df.loc[i, 'answer'], test_df.loc[i, 'attribute'], test_df.loc[i, 'question'])
                ip_prompts.append(ini_prompts['{:s}'.format(test_df.loc[i, 'attribute'])] + prompt)
        else:
            ip_prompts.append('<SKIP>')
    
    # generate response for each prompt 
    all_response = []
    for i, prompt in enumerate(tqdm(ip_prompts)):
        print('Prompt {:d}'.format(i))
        if prompt != '<SKIP>':
            response = run_model(prompt, args, api_keys)
        else:
            response = '<SKIP>'
        all_response.append(response)

    # NOTE: Collect responses
    break_responses = []
    for response in all_response:
        collect_responses = []
        if response == '<SKIP>':
            collect_responses = ['<SKIP>' for _ in range(args.num_samples)]
        else:
            for i in range(args.num_samples):
                response_i = response['choices'][i]['text'].strip()
                collect_responses.append(response_i)
        break_responses.append(collect_responses)
        # print(response['choices']['text'])
    
    break_responses_arr = np.array(break_responses)
    # Dump responses 
    for i in range(args.num_samples):
        test_df['Response_{:d}'.format(i+1)] = break_responses_arr[:, i].tolist()
    
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if args.selective_augmentation:
        sub_dir = 'sel_aug'
        output_dir = os.path.join(output_dir, sub_dir)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
    if args.exim:
        sub_dir = 'ex_im'
        output_dir = os.path.join(output_dir, sub_dir)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
    save_path = os.path.join(output_dir, 'augment_fold_{:d}.csv'.format(args.fold_number))
    test_df.to_csv(save_path, index=False)

if __name__ == '__main__':
    main()
