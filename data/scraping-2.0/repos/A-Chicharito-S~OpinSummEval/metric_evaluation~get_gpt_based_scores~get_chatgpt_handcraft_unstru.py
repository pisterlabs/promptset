import time
import openai
import re
import pandas as pd
import json
from tqdm import tqdm
import os
import sys

# openai.api_base = "https://closeai.deno.dev/v1"
# # openai.api_key = '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'  # mine
# openai.api_key = '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'  # Mason's

openai.api_base = 'https://openai.api2d.net/v1'
openai.api_key = 'fk205563-yEH2DxljhJV1dV1qa518UGTgIeKqYl4H'


processed_cases = 0


# def get_gpt_reply(inst):
#     instruction = 'Evaluate the quality of summaries written for 8 reviews. Rate each summary on 4 dimensions:\n' \
#                   '1. aspect-relevance: Are the mainly discussed aspects in the reviews covered exactly by the summary?\n' \
#                   '2. self-coherence: Is the summary consistent within itself in terms of sentiments and aspects?\n' \
#                   '3. sentiment-consistence: Is the summary consistent with the reviews in terms of sentiments on each aspect?\n' \
#                   '4. readability: Is the summary fluent and informative?\n' \
#                   'You should rate on a scale from 1 (worst) to 5 (best) and reply in separate lines.\n' \
#                   'For each summary, you should begin the reply with the number of the summary, e.g., summary 1\n' \
#                   'For each dimension, you should reply in the format \"dimension: score\", e.g., aspect-relevance: 4\n'
#
#     exemplary_reply = 'An exemplary reply evaluating 3 summaries:\n' \
#                       'summary 1:\naspect-relevance: 3\nself-coherence: 2\nsentiment-consistence: 3\nreadability: 3\n' \
#                       'summary 2:\naspect-relevance: 4\nself-coherence: 5\nsentiment-consistence: 4\nreadability: 4\n' \
#                       'summary 3:\naspect-relevance: 2\nself-coherence: 3\nsentiment-consistence: 1\nreadability: 2\n\n'
#     # system, user, or assistant
#     sys_message = {'role': 'system', 'content': 'You are a helpful assistant'}
#
#     user_message1 = {'role': 'user',
#                      'content': 'Please help me to evaluate 14 summaries over 4 dimensions: '
#                                 '1. aspect-relevance; 2. self-coherence;3. sentiment-consistence; 4. readability.\n'
#                                 'You should rate on a scale from 1 (worst) to 5 (best) and reply in separate lines.\n' \
#                                 'For each summary, you should begin the reply with the number of the summary, e.g., summary 7\n' \
#                                 'For each dimension, you should reply in the format \"dimension: score\", e.g., aspect-relevance: 1\n{}'
#                                 'if you understand the task, please reply \"Yes, I am happy to help\"'.format(
#                          exemplary_reply)}  # TODO: insert the template here
#
#     input_reviews = 'Reviews:\n1.{}\n2.{}\n3.{}\n4.{}\n5.{}\n6.{}\n7.{}\n8.{}\n\n'.format(inst['revs']['rev1'],
#                                                                                           inst['revs']['rev2'],
#                                                                                           inst['revs']['rev3'],
#                                                                                           inst['revs']['rev4'],
#                                                                                           inst['revs']['rev5'],
#                                                                                           inst['revs']['rev6'],
#                                                                                           inst['revs']['rev7'],
#                                                                                           inst['revs']['rev8'])
#     assist_message1 = {'role': 'user', 'content': 'Yes, I am happy to help'}
#
#     input_summs = 'Summaries:\n'
#     model_names = list(inst['model_output'].keys())
#     model_names.sort()
#     assert len(model_names) == 14
#     for model_idx in range(14):
#         input_summs = input_summs + '{}.{}\n'.format(str(model_idx + 1),
#                                                      inst['model_output'][model_names[model_idx]])
#
#     user_message2 = {'role': 'user', 'content': '{}'.format(instruction + input_reviews + input_summs)}
#
#     input_message = [sys_message, user_message1, assist_message1, user_message2]
#
#     completion = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=input_message,
#         temperature=0.0
#     )
#     return completion.choices[0].message


def get_gpt_reply_onebyone(inst):
    instruction = 'Please help me to evaluate the quality of one summary written for 8 reviews. Rate the summary independently on 4 dimensions:\n' \
                  '1. relevance: Are the mainly discussed aspects in the reviews covered exactly by the summary?\n' \
                  '2. coherence: Is the summary consistent within itself in terms of sentiments and aspects?\n' \
                  '3. consistency: Is the summary consistent with the reviews in terms of sentiments for each aspect?\n' \
                  '4. readability: Is the summary fluent and informative?\n' \
                  'You should rate on a scale from 1 (worst) to 5 (best) and evaluate each dimension independently.'
    case = inst['case']
    # system, user, or assistant
    sys_message = {'role': 'system', 'content': 'You are a helpful assistant'}

    user_message1 = {'role': 'user',
                     'content': '{}\n\nif you understand the task, please reply \"Yes, I am happy to help\"'.format(instruction)}  # TODO: insert the template here

    input_reviews = 'Reviews:\n1.{}\n2.{}\n3.{}\n4.{}\n5.{}\n6.{}\n7.{}\n8.{}\n\n'.format(inst['revs']['rev1'],
                                                                                          inst['revs']['rev2'],
                                                                                          inst['revs']['rev3'],
                                                                                          inst['revs']['rev4'],
                                                                                          inst['revs']['rev5'],
                                                                                          inst['revs']['rev6'],
                                                                                          inst['revs']['rev7'],
                                                                                          inst['revs']['rev8'])
    assist_message1 = {'role': 'user', 'content': 'Yes, I am happy to help'}

    model_names = list(inst['model_output'].keys())
    model_names.sort()
    assert len(model_names) == 14

    replies = {}
    for model_idx in range(14):
        model_name = model_names[model_idx]
        input_summs = 'Summary: {}\n'.format(inst['model_output'][model_name])
        user_message = {'role': 'user', 'content': '{}'.format(input_reviews + input_summs)}
        # TODO: another version could be removing the "if-else" structure and treat every reviews as an individual task

        input_message = [sys_message, user_message1, assist_message1, user_message]
        received = False
        while not received:
            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=input_message,
                    temperature=0.0
                )
                time.sleep(1)

                message = completion.choices[0].message
                if model_name not in replies.keys():
                    replies[model_name] = message['content']

                received = True

            except:
                error = sys.exc_info()[0]
                if error == openai.error.InvalidRequestError:
                    # something is wrong: e.g. prompt too long
                    print(f"InvalidRequestError\nInstance & model:\n\n{case+'_'+model_name}\n\n")
                    assert False
                print("API error:", error)
                time.sleep(1)

    return replies


while processed_cases < 100:
    try:
        # ------ data preparation ------
        if os.path.exists('chatgpt_handcraft_score_14_onebyone_unstru.jsonl'):
            f = open('chatgpt_handcraft_score_14_onebyone_unstru.jsonl',
                     'r')  # in case the connection is lost, and you need to run it more than once
            processed_cases = len(f.readlines())
            print(processed_cases)
            f.close()

        model_f = open('../14model_outputs.jsonl', 'r')

        f = open('chatgpt_handcraft_score_14_onebyone_unstru.jsonl', 'a')

        for i, line in tqdm(enumerate(model_f)):
            if i < processed_cases:
                continue
            inst = json.loads(line)
            case = inst['case']
            gpt_reply = get_gpt_reply_onebyone(inst=inst)
            inst_dict = {'gpt_reply': gpt_reply, 'case': int(i + 1)}
            f.write(json.dumps(inst_dict) + '\n')
            print('case {} written'.format(case))
            time.sleep(3)

        f.close()
        processed_cases = case  # should be 100 to end the while loop; if < 100, the program will continue to try
        break
    except Exception:
        time.sleep(3)
