import os
import openai
import json
import random
import time

OPEN_AI_API_KEY = os.environ['OPEN_AI_API_KEY']
openai.api_key = OPEN_AI_API_KEY


def get_completion(data_point):
    print(data_point)
    message = {"role": "user",
               "content": f"""Based on the input in quotation marks, tell me in three sentences what affected this person's decision to become a {', '.join(data_point[1]['occupation'])}.
                        "input": {data_point[1]['early_life'].replace(data_point[0], 'This person')}
                        """}
    chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[message])
    completion = chat_completion.choices[0].message.content
    print(message)
    print(completion)
    data_point[1].update({'gpt_completion': completion})
    return data_point


if __name__ == '__main__':
    with open('data/person_occupation_match_f_multiple.json', 'r') as json_f:
        json_data = json.load(json_f)
    data = random.sample(json_data.items(), 1500)

    completion_list = []
    k = 0
    for i in data:
        k += 1
        print(f'{k}/1500')
        d_p = get_completion(i)
        completion_list.append(d_p)
        if k % 3 == 0:
            time.sleep(60)

    with open('data/person_occupation_gpt3_5_completions.json', 'w') as json_f:
            json.dump(completion_list, json_f, indent=4)
