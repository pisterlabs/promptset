import sys
import json
import openai
import random
import asyncio
import argparse
import platform
from tenacity import retry, stop_after_attempt, wait_random_exponential

sys.path.append('')
from utils import *


if platform.system()=='Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@retry(wait=wait_random_exponential(min=5, max=60), stop=stop_after_attempt(3))
async def generate_response(prompt, engine):
    try:
        if engine == CHAT_GPT or engine == GPT4:
            response = openai.ChatCompletion.create(
                    messages=[{'role': 'user', 'content': prompt}],
                    model=engine,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
        else:
            response = openai.Completion.create(
                    model=engine,
                    prompt=prompt,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=1000,
                )
        return response

    except openai.error.APIError as e:
        if e.status == 429:
            print("Rate limited. Waiting and retrying...")
            asyncio.sleep(1)
        else:
            raise

async def call_api_direct(prompt_list, engine):
    semaphore = asyncio.Semaphore(20)
    tasks = []
    for prompt in prompt_list:
        async with semaphore:
            task = asyncio.create_task(generate_response(prompt, engine))
            tasks.append(task)
    responses = await asyncio.gather(*tasks)
    return responses


@retry(wait=wait_random_exponential(min=5, max=60), stop=stop_after_attempt(5))
async def call_api_improve(prompt_list, engine):
    if engine == CHAT_GPT or engine == GPT4:
        async_responses = [
            openai.ChatCompletion.acreate(
                messages=[{'role': 'user', 'content': prompt}],
                model=engine,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            for prompt in prompt_list
        ]
    else:
        async_responses = [
            openai.Completion.acreate(
                model=engine,
                prompt=prompt,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=1000,
            )
            for prompt in prompt_list
        ]
        
    return await asyncio.gather(*async_responses)


# if random shuffle needed
def shuffle_sents_labels(sents, labels, seed=2):
    combine = list(zip(sents, labels))
    random.seed(seed)
    random.shuffle(combine)
    shuffle_sents, shuffle_labels = zip(*combine)
    return shuffle_sents, shuffle_labels


def LogicNLG_direct(engine, direct_mode, data_path, prompt_path, num_limit, output_path):
    if data_path == None:
        data_path = LOGICNLG_PATH
    data = read_json(data_path=data_path)
    csv_id_list= []
    prompt_list = []

    count = 0
    if prompt_path == None:
        prompt_path = 'prompts/GPT/LogicNLG/prompt_LogicNLG_table2text_%s.txt' %direct_mode
    with open(prompt_path, 'r', encoding='utf-8') as load_file:
        prompt_template = load_file.read()

    for cur_id, current in data.items():
        if count >= num_limit:
            break
        print('Processing Table ' + str(count + 1) + '...')

        csv_id_list.append(current['csv_id'])
        title = current['title']
        table = current['table_text'].replace('<br>', '\n')
    
        prompt = prompt_template.replace('{title}', title).replace('{table}', table)
        prompt_list.append(prompt)
        count += 1

    loop = asyncio.get_event_loop()
    response_list = loop.run_until_complete(call_api_direct(prompt_list=prompt_list, engine=engine))
    
    keywords = ["Claim 1: ", "Claim 2: ", "Claim 3: ", "Claim 4: ", "Claim 5: "]

    write_data = direct_postprocess(engine=engine, table_id_list=csv_id_list, response_list=response_list, keywords=keywords)        

    output_path = get_exact_output_path(output_path=output_path, engine=engine, dataset='LogicNLG', mode='direct', direct_mode=direct_mode)
    json.dump(write_data, open(output_path, "w"), indent=4)


def LogicNLG_improve(engine, finetuned_model_path, prompt_path, num_limit, output_path, num_paths):
    data = read_json(data_path=finetuned_model_path)
    output_data = {}
    count = 0
    if prompt_path == None:
        prompt_path = 'prompts/GPT/LogicNLG/prompt_LogicNLG_improve.txt'
    with open(prompt_path, 'r', encoding='utf-8') as load_file:
        prompt_template = load_file.read()
    for cur_id, current in data.items():
        if count >= num_limit:
            break
        print('Processing Table ' + str(count + 1) + '...')

        title = current['title']
        table = current['table_text'].replace('<br>', '\n')
        csv_id = current['csv_id']
        sentences = current['sentences']
        data[cur_id]['sentences'] = sentences
    
        feedback_list = []
        
        for sent in sentences:
            prompt = prompt_template.replace('{title}', title).replace('{table}', table).replace('{sent}', sent)
            prompt_list = []
            for _ in range(num_paths):
                prompt_list.append(prompt)                  # the same prompt for num_paths times
            
            responses = asyncio.run(call_api_improve(prompt_list=prompt_list, engine=engine))
            
            keywords = ["New claim:", "new claim:","new Claim:","New Claim:", "modified claim:", "modified Claim:" "Modified Claim:", "Modified claim:","Edited Claim:","edited Claim:","edited claim:","Edited claim:", "updated claim:", "Updated claim:","updated Claim:","Updated Claim:"]
            pred_vote = improve_postprocess(engine=engine, responses=responses, keywords=keywords, ori_sent=sent)
                    
            if len(pred_vote['Entailed']) > len(pred_vote['Refuted']):
                feedback_list.append(pred_vote['Entailed'][0].replace('\n',''))
            else:
                feedback_list.append(pred_vote['Refuted'][0].replace('\n',''))
        output_data[csv_id] = feedback_list
        count += 1
    
    finetuned_model_names = ['GPT2', 'r2d2', 't5-base', 'plog-t5-large', 'loft']

    for finetuned in finetuned_model_names:
        if finetuned in finetuned_model_path:
            finetuned_model = finetuned
            break
    
    output_path = get_exact_output_path(output_path=output_path, engine=engine, dataset='LogicNLG', mode='improve', finetuned_model=finetuned_model)
    json.dump(output_data, open(output_path, "w"), indent=4)


def FeTaQA_direct(engine, direct_mode, data_path, prompt_path, num_limit, output_path):
    if data_path == None:
        data_path = FETAQA_PATH
    data = read_json(data_path=data_path)
    feta_id_list= []
    prompt_list = []

    count = 0
    if prompt_path == None:
        prompt_path = 'prompts/GPT/FeTaQA/prompt_FeTaQA_table2text_%s.txt' %direct_mode
    with open(prompt_path, 'r', encoding='utf-8') as load_file:
        prompt_template = load_file.read()

    for cur_id, current in data.items():
        if count >= num_limit:
            break
        print('Processing Table ' + str(count + 1) + '...')

        feta_id_list.append(current['feta_id'])
        page_title = current['page_title']
        section_title = current['section_title']
        question = current['question']
        table = current['table_text'].replace('<br>', '\n')
    
        prompt = prompt_template.replace('{page_title}', page_title).replace('{section_title}', section_title).replace('{question}', question).replace('{table}', table)
        prompt_list.append(prompt)
        count += 1

    loop = asyncio.get_event_loop()
    response_list = loop.run_until_complete(call_api_direct(prompt_list=prompt_list, engine=engine))
    
    keywords = ["Answer: ", "answer: "]

    write_data = direct_postprocess(engine=engine, table_id_list=feta_id_list, response_list=response_list, keywords=keywords)
    
    if 'CoT' in direct_mode:
        write_data = FeTaQA_F2WTQ_CoT_clean(write_data)

    output_path = get_exact_output_path(output_path=output_path, engine=engine, dataset='FeTaQA', mode='direct', direct_mode=direct_mode)
    json.dump(write_data, open(output_path, "w"), indent=4)


def FeTaQA_improve(engine, finetuned_model_path, prompt_path, num_limit, output_path, num_paths):
    answer_data = read_json(data_path=finetuned_model_path)
    table_data = read_json(data_path=FETAQA_PATH)
    output_data = {}
    count = 0

    for cur_id, current in table_data.items():
        if count >= num_limit:
            break
        print('Processing Table ' + str(count + 1) + '...')

        page_title = current['page_title']
        section_title = current['section_title']
        question = current['question']
        table = current['table_text'].replace('<br>', '\n')

        feta_id = str(current['feta_id'])
        answer = answer_data[feta_id]
        if prompt_path == None:
            prompt_path = 'prompts/GPT/FeTaQA/prompt_FeTaQA_improve.txt'
        with open(prompt_path, 'r', encoding='utf-8') as load_file:
            prompt_template = load_file.read()

        prompt = prompt_template.replace('{page_title}', page_title).replace('{section_title}', section_title).replace('{question}', question).replace('{answer}', answer).replace('{table}', table)
        prompt_list = []
        for _ in range(num_paths):
            prompt_list.append(prompt)                  # the same prompt for num_paths times

        responses = asyncio.run(call_api_improve(prompt_list=prompt_list, engine=engine))
        
        keywords = ["New answer:", "new answer:", "new Answer:", "New Answer:", "modified answer:", "modified Answer:", "Modified Answer:", "Modified answer:", "Edited Answer:", "edited Answer:", "edited answer:", "Edited answer:", "updated answer:", "Updated answer:", "updated Answer:", "Updated Answer:"]
        pred_vote = improve_postprocess(engine=engine, responses=responses, keywords=keywords, ori_sent=answer)
                
        if len(pred_vote['Entailed']) > len(pred_vote['Refuted']):
            feedback = pred_vote['Entailed'][0].replace('\n','')

        else:
            feedback = pred_vote['Refuted'][0].replace('\n','')

        output_data[feta_id] = feedback
        count += 1

    finetuned_model_names = ['bart_large', 'flan_t5_large', 'omnitab_large', 'reastap_large', 'tapex_large']

    for finetuned in finetuned_model_names:
        if finetuned in finetuned_model_path:
            finetuned_model = finetuned
            break
    
    output_path = get_exact_output_path(output_path=output_path, engine=engine, dataset='FeTaQA', mode='improve', finetuned_model=finetuned_model)
    json.dump(output_data, open(output_path, "w"), indent=4)


def F2WTQ_direct(engine, direct_mode, data_path, prompt_path, num_limit, output_path):
    if data_path == None:
        data_path = F2WTQ_PATH
    data = read_json(data_path=data_path)
    table_id_list= []
    prompt_list = []

    count = 0
    if prompt_path == None:
        prompt_path = 'prompts/GPT/F2WTQ/prompt_F2WTQ_table2text_%s.txt' %direct_mode
    with open(prompt_path, 'r', encoding='utf-8') as load_file:
        prompt_template = load_file.read()

    for cur_id, current in data.items():
        if count >= num_limit:
            break
        print('Processing Table ' + str(count + 1) + '...')

        table_id_list.append(current['id'])
        question = current['new_question']
        table = current['table_text'].replace('<br>', '\n')
    
        prompt = prompt_template.replace('{question}', question).replace('{table}', table)
        prompt_list.append(prompt)
        count += 1

    loop = asyncio.get_event_loop()
    response_list = loop.run_until_complete(call_api_direct(prompt_list=prompt_list, engine=engine))
    
    keywords = ["Answer: ", "answer: "]

    write_data = direct_postprocess(engine=engine, table_id_list=table_id_list, response_list=response_list, keywords=keywords)
    
    if 'CoT' in direct_mode:
        write_data = FeTaQA_F2WTQ_CoT_clean(write_data)

    output_path = get_exact_output_path(output_path=output_path, engine=engine, dataset='F2WTQ', mode='direct', direct_mode=direct_mode)
    json.dump(write_data, open(output_path, "w"), indent=4)


def LoTNLG_direct(engine, direct_mode, data_path, prompt_path, num_limit, output_path):
    if data_path == None:
        data_path = LOTNLG_PATH
    data = read_json(data_path=data_path)
    csv_id_list= []
    prompt_list = []

    count = 0
    if prompt_path == None:
        prompt_path = 'prompts/GPT/LoTNLG/prompt_LoTNLG_table2text_%s.txt' %direct_mode
    with open(prompt_path, 'r', encoding='utf-8') as load_file:
        prompt_template = load_file.read()

    for cur_id, current in data.items():
        if count >= num_limit:
            break
        print('Processing Table ' + str(count + 1) + '...')

        csv_id_list.append(current['csv_id'])
        title = current['title']
        table = current['table_text'].replace('<br>', '\n')
    
        logical_labels = current['logical_labels']
        logical_labels_str = ''
        for k in range(len(logical_labels)):
            logical_labels_str += f'Logical label {k+1}: {logical_labels[k]}\n'
        prompt = prompt_template.replace('{title}', title).replace('{table}', table).replace('{logical_labels}', logical_labels_str)
        prompt_list.append(prompt)
        count += 1

    loop = asyncio.get_event_loop()
    response_list = loop.run_until_complete(call_api_direct(prompt_list=prompt_list, engine=engine))
    
    keywords = ["Claim 1: ", "Claim 2: ", "Claim 3: ", "Claim 4: ", "Claim 5: "]

    write_data = direct_postprocess(engine=engine, table_id_list=csv_id_list, response_list=response_list, keywords=keywords)        

    output_path = get_exact_output_path(output_path=output_path, engine=engine, dataset='LoTNLG', mode='direct', direct_mode=direct_mode)
    json.dump(write_data, open(output_path, "w"), indent=4)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_org", default=None, type=str, required=True,
                        help='your organization No. for openai')          # org No. for openai
    parser.add_argument("--api_key", default=None, type=str, required=True,
                        help='your api key for openai')                                                     # api key for openai
    parser.add_argument("--engine", default=None, type=str, required=True, 
                        help='which openai engine to use, can select from ["text-davinci-002", "text-davinci-003", "gpt-3.5-turbo", "gpt-4"]')      # openai engine
    parser.add_argument("--temperature", type=float, default=0.7,
                        help='temperature value for openai model, can select from 0 to 2')
    parser.add_argument("--top_p", type=float, default=1.0,
                        help='top_p value for openai model, can select from 0 to 1')
    parser.add_argument("--mode", default=None, type=str, required=True,
                        help='which mode to use, can select from ["direct", "improve"]')
    parser.add_argument("--direct_mode", type=str, default='two_shot_CoT',
                        help='which direct submode to use, only valid when you choose "direct" as mode, can select from ["zero_shot", "one_shot", "two_shot", "one_shot_CoT", "two_shot_CoT"]')
    parser.add_argument("--prompt_path", default=None, type=str,
                        help='which prompt to use, only valid when you choose "direct" as mode. If not specified, we will use the default prompt for each direct mode')
    parser.add_argument("--dataset", default=None, type=str, required=True,
                        help='which dataset to use, can select from ["LogicNLG", "FeTaQA", "F2WTQ", "LoTNLG"]')
    parser.add_argument("--data_path", default=None, type=str,
                        help='original data path for the dataset. If not specified, we will use the default path for each dataset')
    parser.add_argument("--finetuned_model_path", type=str,
                        help='output path of finetuned small model for the dataset, only valid when you choose "improve" as mode')
    parser.add_argument("--output_path", type=str, default="outputs",
                        help='output path for the generated data')
    parser.add_argument("--num_limit", type=int, default=10,
                        help='number of tables used to generate')
    parser.add_argument("--num_paths", type=int, default=3,
                        help='number of paths used to improve using self-consistency mechanism, only valid when you choose "improve" as mode')
    args = parser.parse_args()

    openai.organization = args.api_org
    openai.api_key = args.api_key
    

    if args.dataset == 'LogicNLG':
        if args.mode == 'direct':
            LogicNLG_direct(
                engine=args.engine, 
                direct_mode=args.direct_mode,
                data_path=args.data_path,
                prompt_path=args.prompt_path, 
                num_limit=args.num_limit, 
                output_path=args.output_path
                )

        elif args.mode == 'improve':
            LogicNLG_improve(
                engine=args.engine,
                finetuned_model_path=args.finetuned_model_path,
                prompt_path=args.prompt_path,
                num_limit=args.num_limit, 
                output_path=args.output_path,
                num_paths=args.num_paths
                )
        else:
            raise ValueError("Invalid mode name")
        
    elif args.dataset == 'FeTaQA':
        if args.mode == 'direct':
            FeTaQA_direct(
                engine=args.engine,
                direct_mode=args.direct_mode,
                data_path=args.data_path,
                prompt_path=args.prompt_path,
                num_limit=args.num_limit, 
                output_path=args.output_path
                )
        elif args.mode == 'improve':
            FeTaQA_improve(
                engine=args.engine,
                finetuned_model_path=args.finetuned_model_path,
                prompt_path=args.prompt_path,
                num_limit=args.num_limit, 
                output_path=args.output_path,
                num_paths=args.num_paths
                )
        else:
            raise ValueError("Invalid mode name")
        
    elif args.dataset == 'F2WTQ':
        if args.mode == 'direct':
            F2WTQ_direct(
                engine=args.engine,
                direct_mode=args.direct_mode,
                data_path=args.data_path,
                prompt_path=args.prompt_path,
                num_limit=args.num_limit, 
                output_path=args.output_path
                )
        else:
            raise ValueError("Invalid mode name")
        
    elif args.dataset == 'LoTNLG':
        if args.mode == 'direct':
            LoTNLG_direct(
                engine=args.engine,
                direct_mode=args.direct_mode,
                data_path=args.data_path,
                prompt_path=args.prompt_path,
                num_limit=args.num_limit, 
                output_path=args.output_path
                )
        else:
            raise ValueError("Invalid mode name")
    
    else:
        raise ValueError("Invalid dataset name")