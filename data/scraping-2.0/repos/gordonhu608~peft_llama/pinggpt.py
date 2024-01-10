import argparse
import json
import os

import openai
import tqdm
import ray
import time

openai.api_key = "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"

#os.getenv("OPENAI_API_KEY") echo 
#openai.Model.list()


@ray.remote(num_cpus=4)
def get_eval(max_tokens: int):
        #default conversation templateßß
    system_prompt = "You are an AI visual assistant, and you are seeing a single image. What you see are provided with five sentences, describing the same image you are looking at. Answer all questions as you are seeing the image.\n\
        \n\
        Design a conversation between you and a person asking about this photo. The answers should be in a tone that a visual AI assistant is seeing the image and answering the question.\n\
        Ask diverse questions and give corresponding answers.\n\
        \n\
        Include questions asking about the visual content of the image, including the object types, counting the objects, object actions, object locations, relative positions between objects, etc. Only include questions that have definite answers:\n\
        (1) one can see the content in the image that the question asks about and can answer confidently;\n\
        (2) one can determine confidently from the image that it is not in the image.\n\
        Do not ask any question that cannot be answered confidently.\n\
        \n\
        Also include complex questions that are relevant to the content in the image, for example, asking about background knowledge of the objects in the image, asking to discuss about events happening in the image, etc. Again, do not ask about uncertain details.\n\
        Provide detailed answers when answering complex questions. For example, give detailed examples or reasoning steps to make the content more convincing and well-organized. You can include multiple paragraphs if necessary."
    
    captions = "There is a movie theater that displays the show times above the doors.\n\
        A red fire hydrant is deep in the snow.\n\
        The fire hydrant is in the snow near a recently plowed sidewalk.\n\
        This city has had a very hard winter with snow.\n\
        A hotel for dogs in the snow in winter."
    ques = "What color is the fire hydrant in the image?"
    
    
    while True:
        try:
            response = openai.ChatCompletion.create(
                model= 'gpt-3.5-turbo',  #'gpt-4',
                messages=[{
                    'role': 'system',
                    'content': system_prompt,
                }, {
                    'role': 'user',
                    'content': captions + '\n\n' + ques,
                }],
                temperature=0.2,  # TODO: figure out which temperature is best for evaluation
                max_tokens=max_tokens,
            )
            break
        except openai.error.RateLimitError:
            pass
        except Exception as e:
            print(e)
        time.sleep(1)

    print('success!')
    return response['choices'][0]['message']['content']
    
    
    
# def parse_score(review):
#     try:
#         score_pair = review.split('\n')[0]
#         score_pair = score_pair.replace(',', ' ')
#         sp = score_pair.split(' ')
#         if len(sp) == 2:
#             return [float(sp[0]), float(sp[1])]
#         else:
#             print('error', review)
#             return [-1, -1]
#     except Exception as e:
#         print(e)
#         print('error', review)
#         return [-1, -1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-q', '--question', default ='evalfiles/coco2014_val_gpt4_qa_30x3.jsonl')
    parser.add_argument('-c', '--context', default ='evalfiles/caps_boxes_coco2014_val_80.jsonl') 
    parser.add_argument('-a', '--answer-list', nargs='+', default=['evalfiles/answer_mini7b_trained_01.jsonl', 'evalfiles/answer_self_trained01_Qformer.jsonl']) #'evalfiles/answer_mini7b_trained_01.jsonl answer_self_trained_00'])
    parser.add_argument('-r', '--rule', default = 'prompts/rule.json')
    parser.add_argument('-o', '--output',default = 'evalfiles/qa30x3_review_03_qformer_vs_not.jsonl')
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()

    ray.init()
    
    handle = get_eval.remote(args.max_tokens)
    # To avoid the rate limit set by OpenAI
    time.sleep(1)
    reviews = ray.get(handle)
    print(reviews)
    
    

    # f_q = open(os.path.expanduser(args.question))
    # f_ans1 = open(os.path.expanduser(args.answer_list[0]))
    # f_ans2 = open(os.path.expanduser(args.answer_list[1]))
    # rule_dict = json.load(open(os.path.expanduser(args.rule), 'r'))

    # review_file = open(f'{args.output}', 'w')

    # context_list = [json.loads(line) for line in open(os.path.expanduser(args.context))]
    # image_to_context = {context['image']: context for context in context_list}
    # #print("image_to_context", image_to_context)

    # js_list = []
    # handles = []
    # idx = 0
    # for ques_js, ans1_js, ans2_js in zip(f_q, f_ans1, f_ans2):
    #     ques = json.loads(ques_js)
    #     ans1 = json.loads(ans1_js)
    #     ans2 = json.loads(ans2_js)

    #     inst = image_to_context[ques['image'][13:]] # remove 'COCO_val2014_' prefix

    #     cap_str = '\n'.join(inst['captions'])
    #     box_str = '\n'.join([f'{instance["category"]}: {instance["bbox"]}' for instance in inst['instances']])

    #     category = json.loads(ques_js)['type']
    #     if category in rule_dict:
    #         rule = rule_dict[category]
    #     else:
    #         assert False, f"Visual QA category not found in rule file: {category}."
    #     prompt = rule['prompt']
    #     role = rule['role']
    #     content = (f'[Context]\n{cap_str}\n\n{box_str}\n\n'
    #                f'[Question]\n{ques["instruction"]}\n\n'
    #                f'[{role} 1]\n{ans1["text"]}\n\n[End of {role} 1]\n\n'
    #                f'[{role} 2]\n{ans2["text"]}\n\n[End of {role} 2]\n\n'
    #                f'[System]\n{prompt}\n\n')
    #     js_list.append({
    #         'id': idx+1,
    #         'question_id': ques['id'],
    #         'answer1_id': ans1.get('answer_id', ans1['question_id']),
    #         'answer2_id': ans2.get('answer_id', ans2['answer_id']),
    #         'category': category})
    #     idx += 1
    #     handles.append(get_eval.remote(content, args.max_tokens))
    #     # To avoid the rate limit set by OpenAI
    #     time.sleep(1)

    # reviews = ray.get(handles)
    # for idx, review in enumerate(reviews):
    #     scores = parse_score(review)
    #     js_list[idx]['content'] = review
    #     js_list[idx]['tuple'] = scores
    #     review_file.write(json.dumps(js_list[idx]) + '\n')
    # review_file.close()

