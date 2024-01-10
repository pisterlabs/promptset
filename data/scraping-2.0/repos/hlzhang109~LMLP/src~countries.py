import openai
import numpy as np
import torch
import time
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_utils
import json

if __name__ == '__main__':
    GPU = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(GPU)
    OPENAI_KEY = None  # replace this with your OpenAI API key, if you choose to use OpenAI API

    source = 'huggingface'  # select from ['openai', 'huggingface']
    planning_lm_id = 'gpt2-large'  # see comments above for all options
    translation_lm_id = 'stsb-roberta-large'  # see comments above for all options
    encoder_pooling = 'sentence_embedding'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_STEPS = 10  # maximum number of steps to be generated
    CUTOFF_THRESHOLD = 0.2  # early stopping threshold based on matching score and likelihood score
    P = 0.5  # hyperparameter for early stopping heuristic to detect whether Planning LM believes the plan is finished
    BETA = 0.3  # weighting coefficient used to rank generated samples
    with open('src/countries/test_samples.json', 'r') as f:
        test_samples = json.load(f)
    # create action embeddings using Translated LM
    with open('src/countries/avaliable_rules_r2.json', 'r') as f:
        action_list = json.load(f)

    # create example task embeddings using Translated LM
    with open('src/countries/avaliable_examples_r2.json', 'r') as f:
        available_examples = json.load(f)

    if source == 'openai':
        openai.api_key = OPENAI_KEY
        sampling_params = \
                {
                    "max_tokens": 10,
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
                input_ids = torch.cat([input_prompt_ids, input_entity_ids], dim=1)
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
    # with open('src/countries/test_samples.json', 'r') as f:
    #     test_samples = json.load(f)


    generator = lm_engine(source, planning_lm_id, device)

    # initialize Translation LM
    translation_lm = SentenceTransformer(translation_lm_id).to(device)

    for test in test_samples:
        test = test[6:]
        for example in available_examples:
            if test in example:
                available_examples.remove(example)
        if test in action_list:
            action_list.remove(test)
    action_list_embedding = translation_lm.encode(action_list, batch_size=512, convert_to_tensor=True, device=device, output_value=encoder_pooling)  # lower batch_size if limited by GPU memory

    example_task_list = [example.split('\n')[0] for example in available_examples]  # first line contains the task name
    example_task_embedding = translation_lm.encode(example_task_list, batch_size=512, convert_to_tensor=True, device=device, output_value=encoder_pooling)  # lower batch_size if limited by GPU memory

    def find_most_similar_(query_str, corpus_embedding):
        query_embedding = translation_lm.encode(query_str, convert_to_tensor=True, device=device, output_value=encoder_pooling)

        # calculate cosine similarity against each candidate sentence in the corpus
        cos_scores = st_utils.pytorch_cos_sim(query_embedding, corpus_embedding)[0].detach().cpu().numpy()
        # retrieve high-ranked index and similarity score
        most_similar_idx, matching_score = np.argmax(cos_scores), np.max(cos_scores)
        return most_similar_idx, matching_score

    def find_rule_prompt(predicate, nums, only_rule=False):
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
            steps = av_exp.split('\n')[:-1]
            entities_to_fake, entities = {}, set()
            if len(steps) > 0 and steps[0].split(' ')[2] == predicate:
                num += 1
                for i in range(len(steps)):
                    step = steps[i]
                    if step == steps[0]:
                        _, s, p, o = step.split(' ')
                        example += f'{step}'
                    else:
                        _, _, s, p, o = step.split(' ')
                        example += f'\n {step}'
                    entities.add(s)
                    entities.add(o)
                    
                rules = example
                for idx, k in enumerate(entities):
                    entities_to_fake[k] = chr(idx+ord('A'))
                    rules = rules.replace(k, entities_to_fake[k])

                if only_rule:
                    prompts += rules + '\n\n'
                else:
                    prompts += rules + '\n\n' + example + '\n\n'
                if num == nums:
                    break
        if nums > 1:
            return  rules + prompts +'\n\n'
        else:
            return prompts

    def find_rule_prompt_entity(predicate, nums, only_rule=False):
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
            if av_exp == '':
                continue
            example = ''
            rules = ''
            steps = av_exp.split('\n')[:-1]
            entities_to_fake, entities = {}, set()
            entity = steps[0].split(' ')[1]
            if len(steps) > 0 and entity == predicate:
                num += 1
                for i in range(len(steps)):
                    step = steps[i]
                    if step == steps[0]:
                        _, s, p, o = step.split(' ')
                        example += f'{step}'
                    else:
                        _, _, s, p, o = step.split(' ')
                        example += f'\n {step}'
                    entities.add(s)
                    entities.add(o)
                    
                rules = example
                for idx, k in enumerate(entities):
                    entities_to_fake[k] = chr(idx+ord('A'))
                    rules = rules.replace(k, entities_to_fake[k])

                if only_rule:
                    prompts += rules + '\n\n'
                else:
                    prompts += rules + '\n\n' + example + '\n\n'
                if num == nums:
                    break
        if nums > 1:
            return rules + prompts +'\n\n'
        else:
            return prompts


    # helper function for finding similar sentence in a corpus given a query
    def find_most_similar(query_str, start_entity, corpus_embedding, log_entities = None):
        query_embedding = translation_lm.encode(query_str, convert_to_tensor=True, device=device, output_value=encoder_pooling)

        # calculate cosine similarity against each candidate sentence in the corpus
        cos_scores = st_utils.pytorch_cos_sim(query_embedding, corpus_embedding)[0].detach().cpu().numpy()
        # retrieve high-ranked index and similarity score
        most_similar_idx, matching_score = np.argmax(cos_scores), np.max(cos_scores)
        return most_similar_idx, matching_score

    templates = [
        # "Task: X is located in Z\nStep 1: X is located in Y\nStep 2: Y is located in M\nStep 3: M is located in Z\n,
        #  "Task: X locatedIn Y\nStep 1: X neighborOf M\nStep 2: M neighborOf N\nStep 3: N locatedIn Y\n",
        #  "Task: X locatedIn Y\nStep 1: X neighborOf M\nStep 2: M locatedIn N\nStep 3: N locatedIn Y\n",
        #   "Task: X locatedIn Y\nStep 1: X neighborOf Z\nStep 2: Z locatedIn Y\n",
        "",
        #  "Task: X is located in Z\nStep 1: X is the neighbor of Y\nStep 2: Y is located in M\nStep 3: M is located in Z\n",
        #  "Task: X is located in Z\nStep 1: X is located in Y\nStep 2: Y is the neighbor of M\nStep 3: M is located in Z\n",
    ]
    # define query task
    with open("result.txt","w") as f:
        num_success = [0 for i in range(len(templates))]
        num_steps = np.zeros((len(templates), MAX_STEPS))
        success_tasks = np.zeros((len(test_samples), len(templates)))
        for t in range(len(test_samples)):#
            task = test_samples[t]
            # find most relevant example
            # f.write('*'*30 + ' Test EXAMPLE ' + '-*'*30 + '\n'); 
            # print('*'*40 + ' Test EXAMPLE ' + '*'*40)
            # print(f'{task}')
            for k in range(len(templates)):
                
                # print('*'*40 + ' templates %d ' % (k) + '*'*40)
                example = templates[k]
                start_entity = task.split()[0]
                end_entity = task.split()[-1]
                log_entities = []
                if example == '':
                    example = find_rule_prompt(task.split()[1], 1); print(f'{example}Task: {task}')
                    curr_prompt = f'{example}Task: {task}'
                # construct initial prompt
                curr_prompt = f'{example}\n\n{task}'
                # print example and query task
                # print('-'*10 + ' GIVEN EXAMPLE ' + '-'*10); 
                print(curr_prompt)
                # print('-'*10 + ' EXAMPLE END ' + '-'*10)
                # f.write('-'*10 + ' GIVEN EXAMPLE ' + '-'*10 + '\n'); 
                # f.write(example)
                # f.write('-'*10 + ' EXAMPLE END ' + '-'*10 + '\n')
                # f.write(f'{task}' + '\n')
                for step in range(1, MAX_STEPS + 1):
                    best_overall_score = -np.inf
                    # query Planning LM for single-step action candidates
                    samples, log_probs = generator(curr_prompt + f'\nStep {step}: ', start_entity, sampling_params)
                    for sample, log_prob in zip(samples, log_probs):
                        most_similar_idx, matching_score = find_most_similar(sample, start_entity, action_list_embedding, log_entities)
                        if most_similar_idx == -1 and matching_score == -1:
                            print('[Terminating early no action begin with %s and end with encountered entity\n' % (start_entity))
                            f.write('[Terminating early no action begin with %s and end with encountered entity\n' % (start_entity))
                            break
                        # rank each sample by its similarity score and likelihood score
                        overall_score = matching_score + BETA * log_prob
                        translated_action = action_list[most_similar_idx]
                        if overall_score > best_overall_score:
                            best_overall_score = overall_score
                            best_action = translated_action

                    # terminate early when either the following is true:
                    # 1. top P*100% of samples are all 0-length (ranked by log prob)
                    # 2. overall score is below CUTOFF_THRESHOLD
                    # else: autoregressive generation based on previously translated action
                    top_samples_ids = np.argsort(log_probs)[-int(P * len(samples)):]
                    are_zero_length = all([len(samples[i]) == 0 for i in top_samples_ids])
                    below_threshold = best_overall_score < CUTOFF_THRESHOLD
                    if are_zero_length or most_similar_idx == -1:
                        print(f'\n[Terminating early because top {P*100}% of samples are all 0-length]')
                        f.write(f'\n[Terminating early because top {P*100}% of samples are all 0-length]\n')
                        break
                    else:
                        previous_action = best_action
                        log_entities.append(start_entity)
                        start_entity = previous_action.split()[-1]
                        formatted_action = (best_action[0].lower() + best_action[1:]).replace('_', ' ') # 'open_fridge' -> 'Open fridge'
                        curr_prompt += f'\nStep {step}: {formatted_action}'
                        print(f'Step {step}: {formatted_action}')
                        f.write(f'Step {step}: {formatted_action}\n')
                        if start_entity == end_entity:
                            num_success[k] += 1
                            num_steps[k, step-1] += 1
                            success_tasks[t, k] += 1
                            break

        print("there have %d tasks, and templates are succssful with " % (len(test_samples)), num_success)
        print('The hop counts of the template generation proof are respectively: ', num_steps)
        print('the completion status of each task is', success_tasks)
        print('1example, final success number of tasks is ', np.sum(np.max(success_tasks[:,0:1], axis=1))/len(test_samples))
        print('3example, final success number of tasks is ', np.sum(np.max(success_tasks[:,:3], axis=1)))
        print('5example, final success number of tasks is ', np.sum(np.max(success_tasks[:,:5], axis=1)))
        print('10example, final success number of tasks is ', np.sum(np.max(success_tasks[:,:10], axis=1)))