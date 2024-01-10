import os
import re
import openai
import json
import argparse
import pandas as pd
from tqdm import tqdm

openai.api_key = os.environ.get("OPENAI_API_KEY")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--question_file', type=str, default='./data/question.jsonl', help='path to the question file')
    parser.add_argument('--save_path', type=str, default='./persona', help='path to save the generated persona')
    args = parser.parse_args()
    return args

def load_data():
    data = pd.read_json(args.question_file, lines=True)
    data.turns = data.turns.apply(lambda x: x[0])

    return data

def get_response(question: str, model = "gpt-3.5-turbo-0301") -> str:

    messages=[
    {
        "role": "user",
        "content": f"Two persons with different perspectives are trying to answer the following questions.\nQuestion: {question}\n\nOnly output their job title and self-introduction of the two persons using second perspective in the follow format strictly. Don't need to answer the question.\nCharacter 1: [Your Character Job Title]\nYou are a [...]\n\nCharacter 2: [Your Character Job Title] \nYou are a [...]"
    }
    ]

    response = openai.ChatCompletion.create(
        model = model,
        messages = messages,
        temperature = 0,
        max_tokens = 512,
        top_p = 1,
        frequency_penalty = 0,
        presence_penalty = 0
    )

    return response.choices[0].message['content']

# function to post-process the response into json format
def post_process_response(response: str) -> dict:
    # extract the self-introduction part
    try:
        response = response.split('\n\n')
        # response = [r.replace(':', ' ') for r in response]

        persona = {}
        for res in response:
            if res:
                res = res.strip()
                res = res.split('\n')
                # use re to extract the "[]" part
                who = re.findall(r':(.*)', res[0])[0].strip()
                des = res[1].strip()
                if who:
                    persona[who] = des
    except:
        persona = {'Person 1': '', 'Person 2': ''}

    return persona


if __name__ == '__main__':

    args = get_parser()
    data = load_data()
    response_ls = {"question_id": [], "response": [], "post_process": []}
    for idx, question in tqdm(enumerate(data.turns)):

        response = get_response(question)
        response_ls['question_id'].append(str(data.question_id[idx]))
        response_ls['response'].append(response)
        response_ls['post_process'].append(post_process_response(response))
        print(f'\n---------------- Instance {idx} -----------------')
        print(response)
        print()
        print(post_process_response(response))
        print('-----------------------------------')

    # save the response_ls into json file
    ori_path = os.path.join(args.save_path, 'persona.json')
    post_path = os.path.join(args.save_path, 'persona_post.json')

    with open(ori_path, 'w') as f:
        json.dump(response_ls, f, indent=4, ensure_ascii=False)

    # post-process the json file
    df = pd.read_json(ori_path)
    df = pd.DataFrame(df.post_process)
    df.post_process.tolist().to_json(post_path, orient='records', force_ascii=False, indent=4, lines=True)


