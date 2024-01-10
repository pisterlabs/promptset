import re
import numpy as np
from collections import deque
import cv2
import pandas as pd
import os,sys

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, utils
from PIL import Image, ImageFilter 
from torch import autocast

import PIL
from PIL import Image
import numpy as np
import sklearn
from sklearn.cluster import KMeans
import matplotlib as mpl
import time
from tqdm import tqdm, trange

# matplotlib
import matplotlib.pyplot as plt

from PIL import Image
from transformers import pipeline, CLIPProcessor
# from lavis.models import load_model_and_preprocess

# seaborn
# import seaborn as sns
import openai
import json

# with open(os.path.join(os.path.expanduser('~'),"openai_api_key.txt")) as f:
#     api_key = f.readline().strip()
# openai.api_key = api_key

def display_alongside_batch(img_list, resize_dims=(256,256)):
    res = np.concatenate([np.array(img.resize(resize_dims)) for img in img_list], axis=1)
    return Image.fromarray(res)
# sns.set()


def parse_input(input_str):
    output = {
        'assertions': [],
        'questions': [],
        'visual-verification-probs': [],
        'entities':[],
        'importance-score': [],
        'reason': ''
    }

    lines = input_str.split('\n')
    current_key = None

    for line in lines:
        line = line.strip().lower()
        if not line:
            continue

        if current_key is None and ('assertions:' not in line and 'visual-verification-prob:' not in line and 'questions:' not in line and 'importance-score:' not in line) and 'reason:' not in line:
            continue

        if 'assertions:' in line:
            current_key = 'assertions'
        elif 'visual-verification-prob:' in line:
            current_key = 'visual-verification-probs'
        elif 'questions:' in line:
            current_key = 'questions'
        elif 'entities:' in line:
            current_key = 'entities'
        elif 'importance-score:' in line:
            current_key = 'importance-score'
        elif 'reason:' in line:
            current_key = 'reason'
        elif current_key:
            if current_key == 'visual-verification-probs' or current_key == 'importance-score':
                prob = re.search(r'\d+\.\s*(\d\.\d+)', line)
                if prob:
                    output[current_key].append(float(prob.group(1)))
            elif current_key == 'reason':
                output[current_key] = line
            else:
                point = re.search(r'\d+\.\s*(.+)', line)
                if point:
                    output[current_key].append(point.group(1))

    return output


def parse_input_into_question(input_str):
    output = {
        'assertions': [],
        'questions': []
    }
    lines = input_str.split('\n')
    current_key = None

    for line in lines:
        line = line.strip().lower()
        
        if not line:
            continue

        if current_key is None and ('assertions:' not in line) and ('questions:' not in line):
            continue
        
        if 'assertions:' in line:
            current_key = 'assertions'
        elif 'questions:' in line:
            current_key = 'questions'
        elif current_key:
            point = re.search(r'\d+\.\s*(.+)', line)
            if point:
                output[current_key].append(point.group(1))

    return output


def get_vqa_scores(images, questions, use_neg_scores=False, neg_score_coef=1.0, model_type='blip', device="cuda:0"):
    assert model_type in ['vqa','blip','blip2'], "incorrect model type"
    
    vqa_scores = []
    for k in range(len(images)):
        pos_scores = []
        neg_scores = []
        diff_scores = []
        # print the questions and answers
        for i in range(len(questions)):
            text = questions[i]
            if model_type == 'blip':
                blipvqa_model, blipvqa_vis_processors, blipvqa_txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device='cuda:0')
                image = blipvqa_vis_processors["eval"](images[k]).unsqueeze(0).to(device)
                question = blipvqa_txt_processors["eval"](text)
                with torch.no_grad():
                    answers, vqa_pred = blipvqa_model.predict_answers(samples={"image": image, "text_input": question}, inference_method="rank", answer_list=['yes','no'],num_ans_candidates=2)
                vqa_pred = vqa_pred.cpu()
                pos_scores.append(vqa_pred[0][0])
                if use_neg_scores:
                    neg_scores.append(neg_score_coef*vqa_pred[0][1])
                else: 
                    neg_scores.append(0)
            elif model_type == 'vqa':
                vqa_pipeline = pipeline("visual-question-answering")
                vqa_output = vqa_pipeline(images[k], text, top_k=2)
                vqa_output_ = {}
                for x in vqa_output:
                    vqa_output_[x['answer']] = x['score']#round(x['score'],2)
                try:
                    pos_scores.append(vqa_output_['yes'])
                except:
                    try:
                        pos_scores.append(1-vqa_output_['no'])
                    except:
                        import pdb; pdb.set_trace()
                if 'no' in vqa_output_.keys() and use_neg_scores:
                    neg_scores.append(neg_score_coef*vqa_output_['no'])
                else:
                    neg_scores.append(0.)
            else:
                prompt = "Question: {} Answer:".format(text)
                inputs = blip_processor(images=images[k], text=prompt, return_tensors="pt").to("cpu", torch.float32)
                generated_ids = blip_model.generate(**inputs,)
                generated_text = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                pos_scores.append(get_clip_sentiment_scores(generated_text))
                neg_scores.append(0.)

        diff_scores = np.array(pos_scores)-np.array(neg_scores)
        vqa_scores.append(diff_scores)
    return np.array(vqa_scores)


def DAScore(args, model, transform, images, candidates, device, human_scores, w=2.0):
    len_candidates = [len(c.split()) for c in candidates]
    with open('./zeroshot-imagecaption-eval/decomposable-assertion-gen-prompt-0716') as f:
        message = f.read()
        
    per = []
    parsed_output_res = []

    # with open('parsed_output_res_gpt4_0716_only_question.json', 'r') as f:
    #     parsed_output_res = json.load(f)

    for idx in trange(len(images)):
        image_path = images[idx]
        image = Image.open(image_path)
        
        rating = human_scores[idx]
        caption = candidates[idx]
        message_current = message.format(caption)

        print("begin generating decomposition for ", caption)

        # response = openai.ChatCompletion.create(
        #     model="gpt-4",
        #     messages=[
        #         {"role": "system", "content": "You are a helpful assistant."},
        #         {"role": "user", "content": message_current}
        #     ],
        # )

        # GPT Option2:
        # response = openai.ChatCompletion.create(
        #     model="gpt-4",
        #     # model="gpt-3.5-turbo",
        #     messages=[
        #         {"role": "system", "content": "You are ChatGPT, a model which breaksdown captions of varying complexity into simpler assertions/questions. Answer as concisely as possible."},
        #         {"role": "user", "content": message_current}
        #     ],
        #         temperature=0.2
        # )

        # parsed_output = parse_input(response["choices"][0]["message"]["content"])
        # parsed_output = parse_input_into_question(response["choices"][0]["message"]["content"])
        # parsed_output = parsed_output_res[idx]
        # parsed_output['image_path'] = image_path
        # parsed_output['caption'] = caption
        # print(parsed_output['questions'])
        
        times = 0
        max_times = 5
        while times <= max_times:
            times += 1
            try:
                # GPT Option1:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": message_current}
                    ],
                )

                # GPT Option2:
                # response = openai.ChatCompletion.create(
                #     model="gpt-4",
                #     # model="gpt-3.5-turbo",
                #     messages=[
                #         {"role": "system", "content": "You are ChatGPT, a model which breaksdown captions of varying complexity into simpler assertions/questions. Answer as concisely as possible."},
                #         {"role": "user", "content": message_current}
                #     ],
                #         temperature=0.2
                # )

                # parsed_output = parse_input(response["choices"][0]["message"]["content"])
                parsed_output = parse_input_into_question(response["choices"][0]["message"]["content"])
                parsed_output = parsed_output_res[idx]
                parsed_output['caption'] = caption
                parsed_output['image_path'] = image_path
                print(parsed_output['questions'])
                break
            except:
                time.sleep(30)
                print ("retry, times: %s/%s" % (times, max_times))

            if times > max_times:
                import pdb; pdb.set_trace()

        # calcurate DA score
        # vqa_scores = get_vqa_scores([image], parsed_output['questions'], model_type='vqa', use_neg_scores=True)
        # vqa_scores = get_vqa_scores([image], parsed_output['questions'], model_type='vqa', use_neg_scores=False)
        vqa_scores= get_vqa_scores([image], parsed_output['questions'], model_type='blip', use_neg_scores=True)
        # vqa_scores= get_vqa_scores([image], parsed_output['questions'], model_type='blip', use_neg_scores=False)

        # vqa_score = round(np.mean(vqa_scores) * parsed_output['visual-verification-probs'], 3)
        # vqa_score = round(np.mean(vqa_scores * parsed_output['importance-score']), 3)
        vqa_score = round(np.mean(vqa_scores), 3)
        
        # for visualization
        vqa_scores = [float('{:.3f}'.format(i)) for i in vqa_scores[0]]
        plt.figure(figsize=(10, 15))
        plt.imshow(image)
        # visual_verfication_prob = parsed_output['visual-verification-probs']
        questions = parsed_output['questions']
        # importance = parsed_output['importance-score']
        q_info = ''
        for q in questions[:-1]:
            q_info += q
            q_info += '\n'
        q_info += questions[-1]
        plt.title(f'Caption: {caption}\nHuman-Rating: {rating}\n Vqa-Score: {vqa_score}\n VQA-Scores: {vqa_scores}\n Question: {q_info}')
        plt.axis('off')
        if not os.path.exists('figs/figs_vis/' + args.tag):
            os.makedirs('figs/figs_vis/' + args.tag)
        plt.savefig('figs/figs_vis/' + args.tag + '/visualization_idx' + str(idx) + '_' + image_path.split('/')[-1])

        per.append(vqa_score)
        parsed_output_res.append(parsed_output)

    with open('parsed_output_res_' + args.tag + '.json', 'w') as f:
        json.dump(parsed_output_res, f)

    return np.mean(per), per