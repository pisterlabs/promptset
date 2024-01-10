import os
import json
import openai
import random
from tqdm import tqdm

# Get gpt response
def get_gpt_response_classic(messages: list, model_name='gpt-4', temp=1.0):
    openai.organization = os.environ['OPENAI_KUNLP']
    openai.api_key = os.environ['OPENAI_API_KEY_TIMELINE']

    response = openai.ChatCompletion.create(
        model=model_name,
        temperature=temp,
        messages=messages,
        request_timeout=180
    )

    response_message = response['choices'][0]['message']
    assistant_message = {'role': 'assistant', 'content': response_message['content']}
    messages.append(assistant_message)

    return messages

def get_prompts(example_src: str, tgt: int):
    words_num = 190
    if tgt == 1:
        user_content = f"Summarize the following news article in about {words_num} words without compromising the content.\n"
    elif tgt == 0:
        user_content = f"Summarize the following news article in about {words_num} words or less without compromising the content.\n"
    user_content += example_src
    return user_content

def main():
    with open('/mnt/mint/hara/datasets/news_category_dataset/dataset/diff7_rep1/base/train.json', 'r') as F:
        fake_news_dataset = json.load(F)

    sample_num = 20
    split_string = '[SEP] content: '
    database = {
        'data_from': '/mnt/mint/hara/datasets/news_category_dataset/dataset/diff7_rep1/base/train.json',
        'fake': {
            'raw': {
                'src': [],
                'len': []
            },
            'summary': {
                'src': [],
                'len': []
            }
        },
        'real': {
            'raw': {
                'src': [],
                'len': []
            },
            'summary': {
                'src': [],
                'len': []
            }
        }
    }


    tgt_0_items = [item for item in fake_news_dataset['data'] if item['tgt'] == 0]
    tgt_1_items = [item for item in fake_news_dataset['data'] if item['tgt'] == 1]


    # 'tgt'が0のものと1のものからランダムに10個ずつサンプリング
    # （ただし、もし10個未満のデータしかない場合は、その数だけ取得する）
    sampled_tgt_0 = random.sample(tgt_0_items, min(len(tgt_0_items), sample_num))
    sampled_tgt_1 = random.sample(tgt_1_items, min(len(tgt_1_items), sample_num))

    # 最終的なリストの作成
    sampled_data = sampled_tgt_0 + sampled_tgt_1

    for example in tqdm(sampled_data):
        example_src = example['src'].split(split_string)[1]
        # user_content = get_prompts(example_src, example['tgt'])
        # messages = [{'role': 'user', 'content': user_content}]
        # messages = get_gpt_response_classic(messages, model_name='gpt-4-1106-preview', temp=0.8)
        # res = messages[-1]['content']
        if example['tgt'] == 1:
            database['fake']['raw']['src'].append(example_src)
            database['fake']['raw']['len'].append(len(example_src.split()))
            # database['fake']['summary']['src'].append(res)
            # database['fake']['summary']['len'].append(len(res.split()))

        elif example['tgt'] == 0:
            user_content = get_prompts(example_src, example['tgt'])
            messages = [{'role': 'user', 'content': user_content}]
            messages = get_gpt_response_classic(messages, model_name='gpt-4-1106-preview', temp=0.8)
            res = messages[-1]['content']

            database['real']['raw']['src'].append(example_src)
            database['real']['raw']['len'].append(len(example_src.split()))
            database['real']['summary']['src'].append(res)
            database['real']['summary']['len'].append(len(res.split()))
        else:
            raise ValueError(f'Invalid tgt: {example["tgt"]}')

    ave = lambda x: sum(x) / len(x) if len(x) > 0 else 0
    print('fake')
    print(f'raw: {ave(database["fake"]["raw"]["len"])}, summary: {ave(database["fake"]["summary"]["len"])}')
    print('real')
    print(f'raw: {ave(database["real"]["raw"]["len"])}, summary: {ave(database["real"]["summary"]["len"])}')
    print('==============================================================\n')
    print(database)

    with open(f'/mnt/mint/hara/datasets/news_category_dataset/test/summary_{sample_num}x2_190.json', 'w', encoding='utf-8') as F:
        json.dump(database, F, indent=4, ensure_ascii=False, separators=(',', ': '))

if __name__ == '__main__':
    main()