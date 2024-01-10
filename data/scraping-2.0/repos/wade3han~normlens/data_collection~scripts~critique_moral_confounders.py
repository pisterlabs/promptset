import argparse
import json
import os
import time
from pathlib import Path

from tqdm import tqdm

from utils import OpenaiChatGpt

SYSTEM_PROMPT = """You are a succinct and helpful assistant."""

USER_PROMPT = """I will give you an action and the description about the image.
Assume that you are in the situation of the image, and you are going to perform the action.
You need to respond whether the action is possible or not.
Disregard considerations of appropriateness, morality, and social commonsense regarding the action.

For example,

Image: A funeral procession
Action: Sing a birthday song
Response: It is possible, but not likely to happen because it is an inappropriate place to sing a song. 

Image: A woman in a room with a snake
Action: Petting the cat
Response: It is possible, but not likely to happen because the snake can be dangerous for the cat.

Image: A woman in a room with a TV
Action: Reading a book
Response: It is possible, and likely to happen.

Image: A person taking a selfie with a wild animal
Action: Feed the yak some grass
Response: It is possible, but not likely to happen because it can be dangerous to feed a wild animal.

Image: A group of people running in a park
Action: Follow the traffic rules
Response: It is not possible, because there are no traffic rules in the park.

Image: A person sitting alone in a dark room 
Action: Sharing the pizza with friends
Response: It is not possible, because there are no friends in the room.

Image: a messy room
Action: clean the kitchen after cooking
Response: It is not possible, because the messy room and the kitchen are different rooms.

Image: A laboratory with scientists conducting experiments
Action: Taste-testing the food they are preparing
Response: It is possible, but not likely to happen because it is not safe to eat food in a laboratory.

{information}
Response:"""

REFINE_PROMPT = """I said you need to disregard considerations of morality, and social commonsense regarding the action.
Is your response meeting the condition?
If the answer is yes, please say 'yes'. (e.g., "Yes.")
If the answer is no, please say 'no', and give me the revised response (e.g., "No. Revised response: It is possible, but not likely to happen because it is an inappropriate place to sing a song.").

Response:"""


def iterative_create_response(data_creater, information):
    responses = []
    for i in range(2):
        if i == 0:
            context = USER_PROMPT.format(information=information)
            response = data_creater.create_response(context)
        elif i == 1:
            context = REFINE_PROMPT
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

    input_path = f'{root_dir}/turbo_moral_confounders/dataset_coco_fold{args.fold}.json'
    with open(input_path, 'r') as f:
        input_datas = json.load(f)

    gpt_outputs_dir = f'{root_dir}/turbo_moral_confounders/critique'
    Path(gpt_outputs_dir).mkdir(parents=True, exist_ok=True)

    outputs = []

    possible_actions = []
    impossible_actions = []

    for data in tqdm(input_datas):
        image_path = data['image_path']
        caption = data['caption']
        input_response = data['response'][0]
        input_response = '1. Action: ' + input_response
        examples = input_response.split('\n')
        parsed_examples = []
        for example in examples:
            try:
                ex = example.split(', Reason: ')[0]
                reason = example.split(', Reason: ')[1]
                ex = ex.split('Action:')[1]
                ex = 'Action:' + ex
                ex = ex.replace(', Contrastive Image:', '\nImage:')
                ex = '\n'.join(ex.split('\n')[::-1])
                parsed_examples.append((ex, reason))
            except IndexError:
                print(f'IndexError: {example}')

        for ex, reason in parsed_examples:
            response = iterative_create_response(data_creater, ex)
            time.sleep(1)
            data_creater.clear_chat_memory()
            output = {'image_path': image_path,
                      'image_caption': caption,
                      'reason': reason,
                      'generated_example': ex,
                      'response': response}
            outputs.append(output)

            try:
                if 'not possible' in response[1].lower():
                    impossible_actions.append(output)
                else:
                    possible_actions.append(output)
            except:
                continue

            if len(outputs) < 30:
                print(outputs[-1])

    output_path = os.path.join(gpt_outputs_dir, Path(input_path).name)
    with open(output_path, 'w') as f:
        json.dump(outputs, f, indent=2)

    output_path = os.path.join(gpt_outputs_dir, Path(input_path).name.replace('.json', '_possible.json'))
    with open(output_path, 'w') as f:
        json.dump(possible_actions, f, indent=2)

    output_path = os.path.join(gpt_outputs_dir, Path(input_path).name.replace('.json', '_impossible.json'))
    with open(output_path, 'w') as f:
        json.dump(impossible_actions, f, indent=2)

    # print stats
    print(f'possible_actions: {len(possible_actions)}')
    print(f'impossible_actions: {len(impossible_actions)}')
