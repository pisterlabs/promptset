import os
import copy
import time
import json
import torch
import openai
import numpy as np
from PIL import Image
from lavis.models import load_model_and_preprocess

q_prompt = \
"I have an image. " \
"Ask me questions about the content of this image. " \
"Carefully asking me informative questions to maximize your information about this image content. " \
"Each time ask one question only without giving an answer. " \
"Avoid asking yes/no questions. " \
'Don\'t imagine any contents that are not in the image or answer your own question. ' \
"I'll put my answer beginning with \"Answer:\"." \

sub_q_prompt = \
"Next Question. Avoid asking yes/no questions.\n" \
"Question: "

summary_prompt = \
'Now summarize the information you get in a few sentences. ' \
'Ignore the questions with answers no or not sure. ' \
'Don\'t add information. Don\'t miss information.\n' \
'Summary: '

first_q = 'Describe this image in detail.'

api_key = '<your OpenAI API key>'
openai.api_key = api_key

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)

image_path = 'assets/data/bongard-ow/bongard_ow_test.json'
caption_path = 'chatcap_chatgpt.json'

def chatgpt(chatlog):
    try:
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                messages=[{
                                                    "role": "user",
                                                    "content": chatlog
                                                }],
                                                max_tokens=1024,
                                                temperature=1,
                                                n=1,
                                                frequency_penalty=0,
                                                presence_penalty=0)
        
        text = response['choices'][0]['message']['content'].strip('\n')
        return text
    except:
        print('chatgpt response error')

def chatcap(image):

    question_1 = first_q
    print(f'question_1: {question_1}\n')
    answer_1 = model.generate({"image": image, "prompt": question_1})[0]
    chatlog = f'{q_prompt}\nQuestion: {question_1}\nAnswer: {answer_1}\n{sub_q_prompt}'
    print(chatlog)
    
    question_2 = chatgpt(chatlog)
    print(f'question_2: {question_2}\n')
    answer_2 = model.generate({"image": image, "prompt": question_2})[0]
    chatlog = f"{q_prompt}\nQuestion: {question_1}\nAnswer: {answer_1}\n" \
                f"Question: {question_2}\nAnswer: {answer_2}\n{sub_q_prompt}"
    print(chatlog)

    question_3 = chatgpt(chatlog)
    print(f'question_3: {question_3}\n')
    answer_3 = model.generate({"image": image, "prompt": question_3})[0]
    chatlog = f"{q_prompt}\nQuestion: {question_1}\nAnswer: {answer_1}\n" \
                f"Question: {question_2}\nAnswer: {answer_2}\n" \
                f"Question: {question_3}\nAnswer: {answer_3}\n{sub_q_prompt}"
    print(chatlog)

    question_4 = chatgpt(chatlog)
    print(f'question_4: {question_4}\n')
    answer_4 = model.generate({"image": image, "prompt": question_4})[0]
    chatlog = f"{q_prompt}\nQuestion: {question_1}\nAnswer: {answer_1}\n" \
                f"Question: {question_2}\nAnswer: {answer_2}\n" \
                f"Question: {question_3}\nAnswer: {answer_3}\n" \
                f"Question: {question_4}\nAnswer: {answer_4}\n{sub_q_prompt}"
    print(chatlog)

    question_5 = chatgpt(chatlog)
    print(f'question_5: {question_5}\n')
    answer_5 = model.generate({"image": image, "prompt": question_5})[0]
    chatlog = f"{q_prompt}\nQuestion: {question_1}\nAnswer: {answer_1}\n" \
                f"Question: {question_2}\nAnswer: {answer_2}\n" \
                f"Question: {question_3}\nAnswer: {answer_3}\n" \
                f"Question: {question_4}\nAnswer: {answer_4}\n" \
                f"Question: {question_5}\nAnswer: {answer_5}\n{summary_prompt}"
    print(chatlog)

    summary = chatgpt(chatlog)
    print(f'summary: {summary}')
    return summary

def main():
    captions = []

    with open(image_path, 'r') as f:
        bongard_ow_test = json.load(f)
        for sample in bongard_ow_test:
            uid = sample['uid']
            sample['captions'] = []
            imageFiles = [os.path.join('assets/data/bongard-ow', imageFile) for imageFile in sample['imageFiles']]

            # preprocess the image
            # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
            images = [
                vis_processors["eval"](Image.open(imageFile).convert("RGB")).numpy()
                for imageFile in imageFiles
            ]
            images = torch.from_numpy(np.array(images)).to(device)
            
            for i in range(14):
                image = images[i].unsqueeze(0)
                caption = chatcap(image)
                sample['captions'].append(copy.deepcopy(caption))
            
            captions.append(copy.deepcopy(sample))

            with open(caption_path, "w") as file:
                json.dump(captions, file, indent=4)

if __name__ == '__main__':
    main()