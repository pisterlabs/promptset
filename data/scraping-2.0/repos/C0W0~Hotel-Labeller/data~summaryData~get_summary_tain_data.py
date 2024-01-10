import sys
import os
import json
import openai
from typing import Union

sys.path.append(os.getcwd()+'\\')

from utils import *
from data.labelled.labelling import label

def write_to_local_json(pos_comments: dict[str, list[dict[str, Union[float, str]]]], neg_comments: dict[str, list[dict[str, Union[float, str]]]]) -> None:
    for aspect in ASPECTS_SUMMARY_ALIAS.keys():
        if len(pos_comments) > 0:
            with open(f'./data/summaryData/pos/{aspect}.json', mode='w+') as file_json:
                json.dump(pos_comments[aspect], file_json, indent=2)
        if len(neg_comments) > 0:
            with open(f'./data/summaryData/neg/{aspect}.json', mode='w+') as file_json:
                json.dump(neg_comments[aspect], file_json, indent=2)

def label_using_gpt(config: dict[str, str]):
    test_data = import_random_test_data()

    pos_sentiments: dict[str, list[dict[str, Union[float, str]]]] = {}
    neg_sentiments: dict[str, list[dict[str, Union[float, str]]]] = {}
    for aspect in ASPECTS:
        pos_sentiments[aspect] = []
        neg_sentiments[aspect] = []

    label_result = label(api_key=config['API_KEY'], comment_list=test_data, main_instruction=config['label_instruction'])

    for gpt_resp in label_result:
        try:
            comment: str = gpt_resp['comment']
            for (aspect, score) in gpt_resp['labels'].items():
                sentiment_list: dict[str, list[dict[str, Union[float, str]]]]
                if score > 0.65:
                    sentiment_list = pos_sentiments
                elif score < 0.35:
                    sentiment_list = neg_sentiments
                else:
                    continue

                comment_data = {}
                comment_data['comment'] = comment
                comment_data['score'] = score
                sentiment_list[aspect].append(comment_data)
        except:
            continue

    for (aspect, meta_data) in pos_sentiments.items():
        with open(os.path.join('./data/summaryData/pos', f'{aspect}.json'), mode='w+') as file_json:
            json.dump(meta_data, file_json, indent=2)
    for (aspect, meta_data) in neg_sentiments.items():
        with open(os.path.join('./data/summaryData/neg', f'{aspect}.json'), mode='w+') as file_json:
            json.dump(meta_data, file_json, indent=2)

def use_existing_training_data():
    all_training_data: list[dict[str, Union[str, dict[str, float]]]] = {}
    with open('./data/labelled/neg/labelled.json') as labelled_file:
        all_training_data = json.load(labelled_file)
    with open('./data/labelled/pos/labelled.json') as labelled_file:
        all_training_data.extend(json.load(labelled_file))
    
    pos_sentiments: dict[str, list[dict[str, Union[float, str]]]] = {}
    neg_sentiments: dict[str, list[dict[str, Union[float, str]]]] = {}
    
    for aspect in ASPECTS_SUMMARY_ALIAS.keys():
        pos_sentiments[aspect] = []
        neg_sentiments[aspect] = []

    for comment_data in all_training_data:
        comment: str = comment_data['comment']
        for aspect in ASPECTS_SUMMARY_ALIAS.keys():
            score = comment_data['labels'][aspect]
            if score > 0.65:
                pos_sentiments[aspect].append({ 'comment': comment,  'score': score })
            if score < 0.35:
                neg_sentiments[aspect].append({ 'comment': comment,  'score': score })
            else:
                continue

    for aspect in ASPECTS_SUMMARY_ALIAS.keys():
        with open(f'./data/summaryData/pos/{aspect}.json', mode='r') as file_json:
            pos_sentiments[aspect].extend(json.load(file_json))
        with open(f'./data/summaryData/neg/{aspect}.json', mode='r') as file_json:
            neg_sentiments[aspect].extend(json.load(file_json))
    
    write_to_local_json(pos_sentiments, neg_sentiments)

def filter_comments_gpt(comments: list[str], aspect: str, config: dict[str, str]) -> dict[str, list[str]]:
    api_key = config['API_KEY']

    aspect_keywords = ASPECTS_KEYWORDS[aspect]
    sentiment_keywords = SENTIMENT_KEYWORDS['Generic'] + SENTIMENT_KEYWORDS[aspect]

    to_verify_aspect: list[str] = []
    to_verify_sentiment: list[str] = []
    skip_check: list[str] = []
    for comment in comments:
        unclear_override = False
        for word in UNCLEAR_KEYWORDS:
            if word in comment:
                unclear_override = True
                break
        if unclear_override:
            to_verify_aspect.append(comment)
            continue

        keyword_found = False
        for word in aspect_keywords:
            if word in comment:
                keyword_found = True
                break
        
        if not keyword_found:
            to_verify_aspect.append(f'"{comment}"')
            continue

        keyword_found = False
        for word in sentiment_keywords:
            if word in comment:
                keyword_found = True
                break
        
        if not keyword_found:
            to_verify_sentiment.append(f'"{comment}"')
        else:
            skip_check.append(f'"{comment}"')
    
    openai.api_key = api_key
    decoder = json.JSONDecoder()

    verify_aspect_clusters = cluster_comments(to_verify_aspect)
    instr_aspect = config['filter_instruction_1'].format(ASPECTS_SUMMARY_ALIAS[aspect])

    filtered_result: dict[str, list[str]] = { 'kept': [], 'removed': [] }
    for i in range(len(verify_aspect_clusters)):
        print(f'{i} / {len(verify_aspect_clusters)}')

        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": instr_aspect

                    },
                    {
                        "role": "user",
                        "content": f"[{verify_aspect_clusters[i]}]"
                    },
                ]
            )

            resp: str = completion.choices[0].message.content
            decoded_resp: dict[str, list[str]] = decoder.decode(resp)
            filtered_result['kept'].extend(decoded_resp['kept'])
            filtered_result['removed'].extend(decoded_resp['removed'])

            print(resp)
        except:
            print('request error')
    
    removed_aspect = filtered_result['removed']

    to_verify_sentiment.extend(filtered_result['kept'])
    verify_sentiment_clusters = cluster_comments(to_verify_sentiment)
    instr_sentiment = config['filter_instruction_2'].format(ASPECTS_SUMMARY_ALIAS[aspect])

    filtered_result = { 'kept': [], 'removed': [] }
    for i in range(len(verify_sentiment_clusters)):
        print(f'{i} / {len(verify_sentiment_clusters)}')

        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": instr_sentiment

                    },
                    {
                        "role": "user",
                        "content": f"[{verify_sentiment_clusters[i]}]"
                    },
                ]
            )

            resp: str = completion.choices[0].message.content
            decoded_resp: dict[str, list[str]] = decoder.decode(resp)
            filtered_result['kept'].extend(decoded_resp['kept'])
            filtered_result['removed'].extend(decoded_resp['removed'])

            print(resp)
        except:
            print('request error')
    
    filtered_result['kept'].extend(skip_check)
    filtered_result['removed'].extend(removed_aspect)
    return filtered_result

if __name__ == '__main__':
    config = get_config()

    pos_sentiments: dict[str, list[dict[str, Union[float, str]]]] = {}
    neg_sentiments: dict[str, list[dict[str, Union[float, str]]]] = {}
    for aspect in ASPECTS_SUMMARY_ALIAS.keys():
        pos_sentiments[aspect] = []
        neg_sentiments[aspect] = []
    
    for aspect in ASPECTS_SUMMARY_ALIAS.keys():
        with open(f'./data/summaryData/pos/{aspect}.json', mode='r') as file_json:
            pos_sentiments[aspect].extend(json.load(file_json))
        with open(f'./data/summaryData/neg/{aspect}.json', mode='r') as file_json:
            neg_sentiments[aspect].extend(json.load(file_json))

    for aspect in ASPECTS_SUMMARY_ALIAS.keys():
        comments_pos = list(map(lambda data: data['comment'], pos_sentiments[aspect]))
        comments_neg = list(map(lambda data: data['comment'], neg_sentiments[aspect]))

        filtered = filter_comments_gpt(comments_neg, aspect, config)
        comments_to_remove = set(filtered['removed'])
        neg_sentiments[aspect] = list(filter(lambda data: not data['comment'] in comments_to_remove, neg_sentiments[aspect]))
        filtered = filter_comments_gpt(comments_pos, aspect, config)
        comments_to_remove = set(filtered['removed'])
        pos_sentiments[aspect] = list(filter(lambda data: not data['comment'] in comments_to_remove, pos_sentiments[aspect]))
    
    write_to_local_json(pos_sentiments, neg_sentiments)