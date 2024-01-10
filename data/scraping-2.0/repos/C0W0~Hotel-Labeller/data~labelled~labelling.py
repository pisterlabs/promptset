import json
import os
import openai
import sys
from typing import Union

sys.path.append(os.getcwd()+'\\')

from utils import should_ignore, get_config

def get_train_comments(label_folder: str):
    label_dir = os.path.join(os.getcwd(), 'data', 'labelled', label_folder)
    comment_list: list[str] = []

    for file_name in os.listdir(label_dir):
        if file_name == 'labelled.json': continue
        with open(os.path.join(label_dir, file_name), 'r') as comments_json:
            for comment in json.load(comments_json):
                if should_ignore(comment): continue
                comment_list.append(f"\"{comment}\"")
    return comment_list


def label(api_key: str, comment_list: list[str], main_instruction: str, label_sentiment='none') -> list[dict[str, Union[str, dict[str, float]]]]:
    comment_clusters: list[str] = []
    current_cluster: list[str] = []
    curr_cluster_str_len = 0
    for comment in comment_list:
        current_cluster.append(comment)
        curr_cluster_str_len += len(comment)

        if curr_cluster_str_len >= 2500 or len(current_cluster) >= 10:
            comment_clusters.append(',\n'.join(current_cluster))
            current_cluster.clear()
            curr_cluster_str_len = 0
    
    if curr_cluster_str_len != 0:
        comment_clusters.append(',\n'.join(current_cluster))
        current_cluster.clear()
        curr_cluster_str_len = 0

    # for clustered in comment_clusters:
    #     print(clustered)
    #     print()

    specific_instructions = {'pos': 'The comments are mostly positive, so treat ambiguous sentiments as positive', 'neg': 'The comments are mostly negative, so treat ambiguous sentiments as negative'}
    if label_sentiment in specific_instructions:
        main_instruction += f'\n{specific_instructions[label_sentiment]}'

    openai.api_key = api_key
    decoder = json.JSONDecoder()
    labelled_resp = []

    for i in range(len(comment_clusters)):
        print(f'{i} / {len(comment_clusters)}')

        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": main_instruction

                    },
                    {
                        "role": "user",
                        "content": comment_clusters[i]
                    },
                ]
            )

            resp: str = completion.choices[0].message.content
            labelled_resp.extend(decoder.decode(resp))

            print(resp)
        except:
            print('request failed')
            continue
    
    return labelled_resp


if __name__ == '__main__':
    config = get_config()
    comments = get_train_comments('neg')

    result = label(api_key=config['API_KEY'], comment_list=comments, main_instruction=config['label_instruction'], label_sentiment='neg')
    with open('./data/labelled/neg/labelled.json', 'w+') as json_out:
        json.dump(result, json_out)