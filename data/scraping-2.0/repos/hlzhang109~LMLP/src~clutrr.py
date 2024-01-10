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
argparser.add_argument('--rule_path', action='store', type=str, default='src/clutrr/rules_all.json')
argparser.add_argument('--example_path', action='store', type=str, default='src/clutrr/example_train.json')
argparser.add_argument('--test_path', action='store', type=str, default='src/clutrr/example_test.json')
argparser.add_argument('--planning_lm_id', action='store', type=str, default='gpt2-large')#gpt2-large
argparser.add_argument('--trans_lm_id', action='store', type=str, default='stsb-roberta-large')
args = argparser.parse_args()

print(args)
GPU = args.device
if torch.cuda.is_available():
    torch.cuda.set_device(GPU)
OPENAI_KEY = None  # replace this with your OpenAI API key, if you choose to use OpenAI API

source = 'huggingface'  # select from ['openai', 'huggingface']
planning_lm_id = args.planning_lm_id  # gpt2, gpt2-medium:, gpt2-large, gpt2-xl
translation_lm_id = args.trans_lm_id  # stsb-roberta-base/large, stsb-bert-base/large
encoder_pooling = 'sentence_embedding'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_STEPS = 10  # maximum number of steps to be generated
CUTOFF_THRESHOLD = 0.2  # early stopping threshold based on matching score and likelihood score
P = 0.5  # hyperparameter for early stopping heuristic to detect whether Planning LM believes the plan is finished
BETA = 0.1  # weighting coefficient used to rank generated samples
if source == 'openai':
    openai.api_key = OPENAI_KEY
    sampling_params = \
            {
                "max_tokens": 15,
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
                "max_tokens": 20,
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
            # truncate each sample if it contains '\n' (the current step is finished)
            # e.g. 'open fridge\n<|endoftext|>' -> 'open fridge'
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
    available_examples = json.load(f)

# initialize Translation LM
if args.noisy_rate > 0.:
    with open('src/clutrr/rules_all.json', 'r') as f:
        action_list_noise = json.load(f)
    num_action_noise = len(action_list_noise)
    action_list_noise = action_list_noise[:int(args.noisy_rate * num_action_noise)]

generator = lm_engine(source, planning_lm_id, device)
translation_lm = SentenceTransformer(translation_lm_id).to(device)

if 'test' in args.rule_path:
    action_list_embedding_ = []
    entity_list, relation_list = [], []
    for rules in action_list_:
        if args.noisy_rate > 0.:
            rules.extend(action_list_noise)
        entity_set, relation_set = set(), set()
        action_embedding = translation_lm.encode(rules, batch_size=512, convert_to_tensor=True, device=device, output_value=encoder_pooling)  # lower batch_size if limited by GPU memory
        action_list_embedding_.append(action_embedding)
        for action in rules:
            s, p, _, o = action.split(' ')
            s = s[:-2]
            entity_set.add(s); relation_set.add(p); entity_set.add(o)
        entity_list.append(list(entity_set))
        relation_list.append(list(relation_set))
else:
    entity_set, relation_set = set(), set()
    action_list_embedding_ = translation_lm.encode(action_list_, batch_size=512, convert_to_tensor=True, device=device, output_value=encoder_pooling)  # lower batch_size if limited by GPU memory
    for action in action_list_:
        s, p, _, o = action.split(' ')
        s = s[:-2]
        entity_set.add(s); relation_set.add(p); entity_set.add(o)
    entity_list, relation_list = list(entity_set), list(relation_set)
entity_list_embedding = translation_lm.encode(list(entity_set), batch_size=512, convert_to_tensor=True, device=device, output_value=encoder_pooling)
relation_list_embedding = translation_lm.encode(list(relation_set), batch_size=512, convert_to_tensor=True, device=device, output_value=encoder_pooling)


example_task_list = [example.split('\n')[0] for example in available_examples]  # first line contains the task name
example_task_embedding = translation_lm.encode(example_task_list, batch_size=512, convert_to_tensor=True, device=device, output_value=encoder_pooling)  # lower batch_size if limited by GPU memory

def find_most_similar_(query_str, corpus_embedding, corpus_set, corpus):
    if query_str in corpus_set:
        return query_str
    query_embedding = translation_lm.encode(query_str, convert_to_tensor=True, device=device, output_value=encoder_pooling)

    # calculate cosine similarity against each candidate sentence in the corpus
    cos_scores = st_utils.pytorch_cos_sim(query_embedding, corpus_embedding)[0].detach().cpu().numpy()
    # retrieve high-ranked index and similarity score
    most_similar_idx, matching_score = np.argmax(cos_scores), np.max(cos_scores)
    return corpus[most_similar_idx]

# helper function for finding similar sentence in a corpus given a query
def find_most_similar(query_str, start_entity, task, corpus_embedding, log_entities = None, action_list=None):
    if start_entity != '' and start_entity != ' ':
        corpus_embedding_, ids = [], []
        for i in range(len(action_list)):
            s = action_list[i].split("'")[0]
            o = action_list[i].split(" ")[-1]
            if s == start_entity and o not in log_entities and action_list[i] != task:
                corpus_embedding_.append(corpus_embedding[i].unsqueeze(0))
                ids.append(i)
        if len(ids) == 0:
            return -1, -1
        corpus_embedding_ = torch.cat(corpus_embedding_).to(device)
    else:
        corpus_embedding_ = corpus_embedding
    query_embedding = translation_lm.encode(query_str, convert_to_tensor=True, device=device, output_value=encoder_pooling)

    # calculate cosine similarity against each candidate sentence in the corpus
    cos_scores = st_utils.pytorch_cos_sim(query_embedding, corpus_embedding_)[0].detach().cpu().numpy()
    # retrieve high-ranked index and similarity score
    most_similar_idx, matching_score = np.argmax(cos_scores), np.max(cos_scores)
    if start_entity != '' and start_entity != ' ':
        most_similar_idx = ids[most_similar_idx]
        del corpus_embedding_
    return most_similar_idx, matching_score    

def find_rule_prompt(predicate, nums, only_rule=False, reverse_rule=False):
    '''
        predicate: the target predicate
        nums: numbers of templates that needs to find
    '''

    num = 0
    prompts  = ''
    available_examples_t = available_examples
    np.random.shuffle(available_examples_t)
    for i in range(len(available_examples_t)):
        av_exp = available_examples_t[i]
        example = ''
        rules = ''
        steps = av_exp.split('\n')
        entities_to_fake, entities = {}, set()
        if steps[0].split(' ')[1] == predicate:
            num += 1
            for i in range(len(steps)):
                step = steps[i]
                s, p, _, o = step.split(' ')
                s = s[:-2]
                entities.add(s)
                entities.add(o)
                if step == steps[0]:
                    example += f'Task: {step}'
                else:
                    example += f'\n Step {i}: {step}'
            
            rules = example
            for idx, k in enumerate(entities):
                entities_to_fake[k] = chr(idx+ord('A'))
                rules = rules.replace(k, entities_to_fake[k])
            
            if only_rule:
                return rules + '\n\n' 
            if reverse_rule:
                prompts += example + '\n\n' + rules + '\n\n'
            else:
                prompts += rules + '\n\n' + example + '\n\n'
            if num == nums:
                break
    if nums > 1:
        return prompts + rules +'\n\n'
    else:
        return prompts

def find_rule_prompt_entity(entity, nums, only_rule=False):
    '''
        entity: the target entity
        nums: numbers of templates that needs to find
    '''

    num = 0
    prompts  = ''
    available_examples_t = available_examples
    np.random.shuffle(available_examples_t)
    for i in range(len(available_examples_t)):
        av_exp = available_examples_t[i]
        example = ''
        rules = ''
        steps = av_exp.split('\n')
        entities_to_fake, entities = {}, set()
        if steps[0].split(' ')[0][:-2] == entity:
            num += 1
            for i in range(len(steps)):
                step = steps[i]
                s, p, _, o = step.split(' ')
                s = s[:-2]
                entities.add(s)
                entities.add(o)
                if step == steps[0]:
                    example += f'Task: {step}'
                else:
                    example += f'\n Step {i}: {step}'
            
            rules = example
            for idx, k in enumerate(entities):
                entities_to_fake[k] = chr(idx+ord('A'))
                rules = rules.replace(k, entities_to_fake[k])
            
            if only_rule:
                return rules + '\n\n' 
            prompts += rules + '\n\n' + example + '\n\n'
            if num == nums:
                break
    if nums > 1:
        return prompts + rules +'\n\n'
    else:
        return prompts

def find_random_prompt(nums, only_rule=False):
    '''
        entity: the target entity
        nums: numbers of templates that needs to find
    '''

    num = 0
    prompts  = ''
    available_examples_t = available_examples
    np.random.shuffle(available_examples_t)
    for i in range(nums):
        av_exp = available_examples_t[i]
        example = ''
        rules = ''
        steps = av_exp.split('\n')
        entities_to_fake, entities = {}, set()
        for i in range(len(steps)):
            step = steps[i]
            s, p, _, o = step.split(' ')
            s = s[:-2]
            entities.add(s)
            entities.add(o)
            if step == steps[0]:
                example += f'Task: {step}'
            else:
                example += f'\n Step {i}: {step}'
        
        rules = example
        for idx, k in enumerate(entities):
            entities_to_fake[k] = chr(idx+ord('A'))
            rules = rules.replace(k, entities_to_fake[k])
        
        if only_rule:
            return rules + '\n\n' 
        prompts += rules + '\n\n' + example + '\n\n'
    if nums > 1:
        return prompts + rules +'\n\n'
    else:
        return prompts
templates = [
    "" for i in range(1)
]
# define query task
with open("result_3_templates.txt","w") as f:
    names = ['1.5', '1.6', '1.7', '1.8', '1.9', '1.10']
    for k in range(len(all_test_samples)):
        print('*'*40 + '\t Clutrr' + str(names[k]) + '\t' + '*'*40)
        if 'test' in args.rule_path:
            action_list_embedding = action_list_embedding_[k]
            action_list = action_list_[k]
        else:
            action_list_embedding = action_list_embedding_
            action_list = action_list_
        test_samples = all_test_samples[k]
        num_success = [0 for i in range(len(templates))]
        num_steps = np.zeros((len(templates), MAX_STEPS))
        success_tasks = np.zeros((len(test_samples), len(templates)))
        for t in range(len(test_samples)):
            task = test_samples[t].split('\n')[0]
            print('*'*20 + f' {task} ' + '*'*20)
            for k in range(len(templates)):
                print('*'*10 + ' templates %d ' % (k) + '*'*10)
                example = ''
                start_entity = task.split("'")[0]
                end_entity = task.split()[-1]
                log_entities = []
                curr_string = f'{task}, '
                curr_prompt = f'Task: {task}'
                if example == '':
                    example = find_rule_prompt(task.split()[1], nums=args.num_rule, only_rule=False, reverse_rule=True);  
                    if not args.salient:
                        print(f'{example}Task: {task}')
                    curr_prompt = f'{example}Task: {task}'
                for step in range(1, MAX_STEPS + 1):
                    best_overall_score = -np.inf
                    samples, log_probs = generator(curr_prompt + f'\n Step {step}: ', start_entity=' ', sampling_params=sampling_params)
                    for sample, log_prob in zip(samples, log_probs):
                        most_similar_idx, matching_score = find_most_similar(sample, start_entity, task, action_list_embedding, log_entities, action_list=action_list)
                        if most_similar_idx == -1 and matching_score == -1:
                            print('[Terminating early no action begin with %s and end with encountered entity\n' % (start_entity))
                            break
                        # rank each sample by its similarity score and likelihood score
                        overall_score = matching_score + BETA * log_prob
                        translated_action = action_list[most_similar_idx]
                        if overall_score > best_overall_score:
                            best_overall_score = overall_score
                            best_action = translated_action
                    top_samples_ids = np.argsort(log_probs)[-int(P * len(samples)):]
                    are_zero_length = all([len(samples[i]) == 0 for i in top_samples_ids])
                    below_threshold = best_overall_score < CUTOFF_THRESHOLD
                    if are_zero_length or most_similar_idx == -1:
                        print(f'\n[Terminating early because top {P*100}% of samples are all 0-length]')
                        break
                    else:
                        previous_action = best_action
                        start_entity = previous_action.split()[-1]
                        log_entities.append(start_entity)
                        formatted_action = (best_action[0] + best_action[1:]).replace('_', ' ').replace('-', ' ')  # 'open_fridge' -> 'Open fridge'
                        curr_prompt += f'\nStep {step}: {formatted_action}'
                        curr_string += (best_action[0] + best_action[1:]) + ', '
                        if not args.salient:
                            print(f'Step {step}: {formatted_action}')
                        if start_entity == end_entity:
                            num_success[k] += 1
                            num_steps[k, step - 1] += 1
                            success_tasks[t, k] += 1
                            print(curr_prompt)
                            f.write(f'{curr_string[:-2]}\n')
                            break
        print("there have %d tasks, and templates are succssful with " % (len(test_samples)), num_success)
        f.write("there have %d tasks, and templates are succssful with\n" % (len(test_samples)))
        f.writelines(str(num_success))

        print('The hop counts of the template generation proof are respectively: ', num_steps)
        f.write('\nThe hop counts of the template generation proof are respectively:\n' % (num_steps))
        f.writelines(str(num_steps))

        print('the completion status of each task is', success_tasks)
        f.write('\nthe completion status of each task is\n')
        f.writelines(str(success_tasks))

        print('final success number of tasks is ', np.sum(np.max(success_tasks, axis=1)))
        f.write('\nfinal success number of tasks is %d' % (np.sum(np.max(success_tasks, axis=1))))