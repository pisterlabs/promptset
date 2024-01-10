import argparse
import json
import os
import time
from pathlib import Path

from tqdm import tqdm

from utils import OpenaiChatGpt

SYSTEM_PROMPT = """You are a succinct and moral assistant."""

USER_PROMPT = """I will give you an action and the description about the image.
Assume that you are in the situation of the image, and you are going to perform the action.
Your goal is to respond whether the action is morally inappropriate or not.

For example,

Image: A funeral procession
Action: Sing a birthday song
Response: It is morally inappropriate to perform the action, because it is an inappropriate place to sing a song.

Image: A woman in a room with a snake
Action: Petting the cat
Response: It is morally inappropriate, because it is not safe to pet the cat if the snake is in the room.

Image: A woman in a room with a TV
Action: Reading a book
Response: It is morally appropriate, because it is okay to read a book in the room. 

Image: A person taking a selfie with a wild animal
Action: Feed the yak some grass
Response: It is morally inappropriate, because it is not safe to feed the wild animal.

Image: A laboratory with scientists conducting experiments
Action: Opening the window
Response: It is morally inappropriate, because the dangerous chemicals might leak out of the window.

{information}
Response:"""


def iterative_create_response(data_creater, information):
    responses = []
    for i in range(1):
        if i == 0:
            context = USER_PROMPT.format(information=information)
            response = data_creater.create_response(context)
        else:
            raise NotImplementedError

        time.sleep(1)
        responses.append(response)
    return responses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--root-dir', type=str, required=True)
    args = parser.parse_args()

    data_creater = OpenaiChatGpt(engine='gpt-3.5-turbo',
                                 temperatue=0.1,
                                 topp=0.95,
                                 frequency_penalty=0.0,
                                 presence_penalty=0.0)
    root_dir = args.root_dir
    data_creater.set_system_prompt(SYSTEM_PROMPT)

    input_path = f'{args.root_dir}/turbo_moral_confounders/critique/dataset_coco_fold{args.fold}_possible.json'
    with open(input_path, 'r') as f:
        input_datas = json.load(f)

    gpt_outputs_dir = f'{root_dir}/turbo_moral_confounders/critique/v3'
    Path(gpt_outputs_dir).mkdir(parents=True, exist_ok=True)

    morally_inappropriate = []
    morally_appropriate = []

    for i, data in tqdm(list(enumerate(input_datas))):
        generated_example = data['generated_example']
        response = iterative_create_response(data_creater, generated_example)
        time.sleep(1)
        data_creater.clear_chat_memory()
        input_datas[i]['moral_judgment'] = response[0]

        if 'morally inappropriate' in response[0] and 'not morally inappropriate' not in response[0]:
            morally_inappropriate.append(data)
            print(data['image_caption'])
            print(data['generated_example'])
            print(data['moral_judgment'])
            print('----')
        else:
            morally_appropriate.append(data)

    output_path = os.path.join(gpt_outputs_dir, Path(input_path).name)
    with open(output_path.replace('.json', '_moral.json'), 'w') as f:
        json.dump(input_datas, f, indent=2)

    # print stats
    print(f'Number of morally inappropriate: {len(morally_inappropriate)}')
    print(f'Number of morally appropriate: {len(morally_appropriate)}')
