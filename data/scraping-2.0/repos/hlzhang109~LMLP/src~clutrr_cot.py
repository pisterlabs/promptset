from selectors import EpollSelector
import openai
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_utils
import json
import argparse

argparser = argparse.ArgumentParser('CLUTRR', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument('--device', action='store', type=int, default=1)
argparser.add_argument('--num_rule', action='store', type=int, default=1)
argparser.add_argument('--noisy_rate', action='store', type=float, default=0.0)
argparser.add_argument('--salient', type=bool, default=False)
argparser.add_argument('--train_id', type=int, default=0)
argparser.add_argument('--rule_path', action='store', type=str, default='src/clutrr/rules_all.json')
argparser.add_argument('--example_path', action='store', type=str, default='src/clutrr/example_clutrr_train_story.json')
argparser.add_argument('--test_path', action='store', type=str, default='src/clutrr/example_clutrr_test_story.json')
argparser.add_argument('--planning_lm_id', action='store', type=str, default='text-davinci-002')#gpt2-large
argparser.add_argument('--trans_lm_id', action='store', type=str, default='stsb-roberta-large')
args = argparser.parse_args()

print(args)
GPU = args.device
if torch.cuda.is_available():
    torch.cuda.set_device(GPU)
OPENAI_KEY = None # replace this with your OpenAI API key, if you choose to use OpenAI API

source = 'openai'  # select from ['openai', 'huggingface']
planning_lm_id = args.planning_lm_id  # gpt2, gpt2-medium:, gpt2-large, gpt2-xl
encoder_pooling = 'sentence_embedding'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_STEPS = 1  # maximum number of steps to be generated
CUTOFF_THRESHOLD = 0.2  # early stopping threshold based on matching score and likelihood score
P = 0.5  # hyperparameter for early stopping heuristic to detect whether Planning LM believes the plan is finished
BETA = 0.1  # weighting coefficient used to rank generated samples
if source == 'openai':
    openai.api_key = OPENAI_KEY
    sampling_params = \
            {
                "max_tokens": 100,
                "temperature": 0.6,
                "top_p": 0.9,
                "n": 10,
                "logprobs": 1,
                "presence_penalty": 0.5,
                "frequency_penalty": 0.3,
                "stop": '\n'
            }
elif source == 'huggingface':
    sampling_params = \
            {
                "max_tokens": 100,
                "temperature": 0.1,
                "top_p": 0.9,
                "num_return_sequences": 10,
                "repetition_penalty": 1.2,
                'use_cache': True,
                'output_scores': True,
                'return_dict_in_generate': True,
                'do_sample': True,
            }

def lm_engine(source, planning_lm_id, device):
    if source == 'huggingface':
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(planning_lm_id)
        model = AutoModelForCausalLM.from_pretrained(planning_lm_id, pad_token_id=tokenizer.eos_token_id).to(device)

    def _generate(prompt, start_entity, sampling_params):
        if source == 'openai':
            response = openai.Completion.create(engine=planning_lm_id, prompt=prompt, **sampling_params)
            generated_samples = [response['choices'][i]['text'] for i in range(sampling_params['n'])]
            # calculate mean log prob across tokens
            mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in range(sampling_params['n'])]
        elif source == 'huggingface':
            input_prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            input_entity_ids = tokenizer(start_entity, return_tensors="pt").input_ids.to(device)
            prompt_len = input_prompt_ids.shape[-1]; entity_len = input_entity_ids.shape[-1]
            if start_entity != ' ':
                input_ids = torch.cat([input_prompt_ids, input_entity_ids], dim=1)
            else:
                input_ids = input_prompt_ids; entity_len = 0

            output_dict = model.generate(input_ids, max_length=prompt_len + sampling_params['max_tokens'], **sampling_params)
            # discard the prompt (only take the generated text)
            generated_samples = tokenizer.batch_decode(output_dict.sequences[:, prompt_len:])
            # calculate per-token logprob
            vocab_log_probs = torch.stack(output_dict.scores, dim=1).log_softmax(-1)  # [n, length, vocab_size]
            token_log_probs = torch.gather(vocab_log_probs, 2, output_dict.sequences[:, prompt_len + entity_len:, None]).squeeze(-1).tolist()  # [n, length]
            for i, sample in enumerate(generated_samples):
                stop_idx = sample.index('\n') if '\n' in sample else None
                generated_samples[i] = sample[:stop_idx]
                token_log_probs[i] = token_log_probs[i][:stop_idx]
            # calculate mean log prob across tokens
            mean_log_probs = [np.mean(token_log_probs[i]) for i in range(sampling_params['num_return_sequences'])]
        generated_samples = [sample.strip().lower() for sample in generated_samples]
        return generated_samples, mean_log_probs

    return _generate

# create example task embeddings using Translated LM
with open(args.test_path, 'r') as f:
    all_test_samples = json.load(f)

# create action embeddings using Translated LM
with open(args.rule_path, 'r') as f:
    action_list_ = json.load(f)

# create example task embeddings using Translated LM
with open(args.example_path, 'r') as f:
    available_examples_ = json.load(f)

generator = lm_engine(source, planning_lm_id, device)

def find_rule_prompt(available_examples, task = '', predicate='', nums=1, leng=0):
    '''
        predicate: the target predicate
        nums: numbers of templates that needs to find
    '''

    num = 0
    prompts  = ''
    available_examples_t = available_examples
    
    np.random.shuffle(available_examples_t)
    example_task_list_t = [example[1] for example in available_examples]
    for i in range(len(available_examples_t)):
        
        if predicate != '' and example_task_list_t[i] != predicate or task[:40] == available_examples_t[i][0][:40]:
            continue
        triplets = available_examples_t[i][0].split('\n')[-1].split(',')
        if leng <= 4 and len(triplets) != leng:
            continue
        prompts += available_examples_t[i][0] + '\n\n'
        
        num += 1
        if num == nums:
            break
    return prompts

templates = [
    "" for i in range(1)
]
# define query task
test_relation_lst = ["aunt", "brother", "daughter", "daughter-in-law", "father", "father-in-law", "granddaughter",
                         "grandfather", "grandmother", "grandson", "mother", "mother-in-law", "nephew", "niece",
                         "sister", "son", "son-in-law", "uncle", "husband", "wife"]
relation_value = [1, 0, -1, -1, 1, 1, -2, 2, 2, -2, 1, 1, -1, -1, 0, -1, -1, 1, 0, 0]
relation_sex = [0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0]
all_success = np.zeros((len(all_test_samples), len(all_test_samples)))
relation_to_idx = {test_relation_lst[r]:r for r in range(len(test_relation_lst))}

for a in range(0, len(all_test_samples)):#len(available_examples_)
    names = ['1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8', '1.9', '1.10']
    idx = [0, 0, 0, 1, 2, 3, 4, 5, 6]
    print('*'*40 + '\t Rule from' + str(names[a]) + '\t' + '*'*40)
    available_examples = available_examples_[idx[a]]
    
    for k in range(len(all_test_samples)):
        print('*'*30 + '\t Clutrr' + str(names[k]) + '\t' + '*'*30)
        test_samples = all_test_samples[k]
        success_tasks = 0
        for t in range(len(test_samples)):
            
            task, relation = test_samples[t]
            print('*'*20 + test_samples[t][0].split('.')[-1][:-10]  + '*'*20)
            for _ in range(len(templates)):
                print('*'*10 + ' templates %d ' % (_) + '*'*10)
                example = ''
                start_entity = task.split("?")[0].split(' ')[-1]
                end_entity = task.split()[-1]
                log_entities = []
                curr_prompt = f'{task}'
                if example == '':
                    example = find_rule_prompt(available_examples.copy(), task=task, predicate='', nums=args.num_rule, leng=a+2); 
                    curr_prompt = f'{example}{task}' 
                   
                    if not args.salient:
                        print(curr_prompt, end=' ')

                for step in range(1, MAX_STEPS + 1):
                    # start_entity = ' '
                    samples, log_probs = generator(curr_prompt, start_entity=start_entity, sampling_params=sampling_params)
                    # best_id = np.array(log_probs).argmax()
                    # r = samples[best_id].split(' ')[-1][:-1]
                    # print(samples[best_id])
                    best_id = (-np.array(log_probs)).argsort()
                    for id in best_id:
                        if '' != samples[id]:
                            print(samples[id])
                            r = samples[id].split(' ')[-1][:-1]
                            break
                    else:
                        r = ''
                    
                    print(f"task relation: {relation}")
                    if r.lower() in test_relation_lst and relation.lower() in test_relation_lst:
                        id_r = relation_to_idx[r.lower()]
                        id = relation_to_idx[relation.lower()]
                        if relation_value[id_r] == relation_value[id] or relation_value[id_r] == -relation_value[id]:
                            success_tasks += 1
    
        print("there have %d tasks, and templates are succssful with " % (len(test_samples)), success_tasks)
        all_success[a][k] = success_tasks
        print(all_success)

print(all_success)
