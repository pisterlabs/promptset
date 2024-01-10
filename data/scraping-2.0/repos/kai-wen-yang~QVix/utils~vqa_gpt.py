import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb
from .tools import VQAEval
import pdb
import sys
sys.path.append("..")
from models import get_image
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  
from .prompt import prompt_template
import collections
from datasets import Dataset, load_dataset

@retry(wait=wait_random_exponential(min=0.1, max=0.2), stop=stop_after_attempt(10))
def call_gpt(chatgpt_messages, key, model="gpt-3.5-turbo", temp_gpt=0.0):
    openai.api_key = key
    response = openai.ChatCompletion.create(model=model, messages=chatgpt_messages, temperature=temp_gpt, max_tokens=128)
    reply = response['choices'][0]['message']['content']
    total_tokens = response['usage']['total_tokens']
    return reply, total_tokens


def evaluate_VQA_gpt(
    model,
    dataset,
    model_name,
    dataset_name,
    time,
    batch_size=1,
    answer_path='./answers',
    max_new_tokens=256,
    args=None,
    classnames=None
):  
    vqa_gpt_prompt = prompt_template[args.prompt]
    predictions=[]
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})
    for t, batch in enumerate(tqdm(dataloader, desc="Running inference")):

        questions = []
        subquestions = []
        options = []
        for i in range(len(batch['image_path'])):
            if 'subquestion' in dataset[0].keys():
                reply = batch['subquestion'][i]
            else:
                message = [{'role': 'user', 'content': vqa_gpt_prompt.format(f"{batch['question'][i]}\n")}]
                reply, total_tokens = call_gpt(message, args.api_key)
            subquestions.append(reply)
            question = reply+'\n'+args.question.format(batch['question'][i])
            questions.append(question)
            option = batch['question'][i].split('Options:')[1].split('\n')[:-1]
            options.append(option)

        outputs = model.batch_generate(batch['image_path'], questions, max_new_tokens=max_new_tokens)
        for image_path, question, gt_answer, output, subject, grade, subque, option in zip(batch['image_path'], questions, batch['gt_answers'], outputs, batch['subject'], batch['grade'], subquestions, options):
            split_output = output
            answer_dict={'question': question, 'answer': split_output, 'ori_answer': output,
            'gt_answers': gt_answer, 'image_path': image_path, 'option': option,
            'model_name': model_name, 'subject': subject, 'grade': grade}
            predictions.append(answer_dict)

        text_table = wandb.Table(columns=["answer", "label",  "question", "ori_answer"])
        if type(gt_answer) is not str:
            gt_answer = gt_answer[0]
        text_table.add_data(split_output, gt_answer, question, output)
        wandb.log({f'time{time}_batch{str(t)}/image.jpg': wandb.Image(get_image(image_path)),
                   f'time{time}_batch{str(t)}/table': text_table,
                   })

    answer_dir = os.path.join(answer_path, time)
    os.makedirs(answer_dir, exist_ok=True)
    answer_path = os.path.join(answer_dir, f"{dataset_name}.json")
    with open(answer_path, "w") as f:
        f.write(json.dumps(predictions, indent=4))
    eval = VQAEval()
    correct, correct_nat, correct_soc, correct_lan, correct_g16, correct_g712 = 0, 0, 0, 0, 0, 0
    num, num_nat, num_soc, num_lan, num_g16, num_g712 = 0, 0, 0, 0, 0, 0
    with open(answer_path, 'r') as f:
        dict = json.load(f)
        for i in range(len(dict)):
            gt_answers = dict[i]['gt_answers']
            answer = dict[i]['answer']
            hallu = 0
            for opt in dict[i]['option']:
                hallu+=eval.evaluate(answer, opt)
            if eval.evaluate(answer, gt_answers)==1:
                correct+=1
                if dict[i]['subject'] == 'natural science':
                    correct_nat+=1
                    num_nat+=1
                if dict[i]['subject'] == 'social science':
                    correct_soc+=1
                    num_soc+=1
                if dict[i]['subject'] == 'language science':
                    correct_lan+=1
                    num_lan+=1
                if dict[i]['grade'] in ['grade1', 'grade2', 'grade3', 'grade4', 'grade5', 'grade6']:
                    correct_g16+=1
                    num_g16+=1
                if dict[i]['grade'] in ['grade7', 'grade8', 'grade9', 'grade10', 'grade11', 'grade12']:
                    correct_g712+=1
                    num_g712+=1
            else:
                if dict[i]['subject'] == 'natural science':
                    num_nat+=1
                if dict[i]['subject'] == 'social science':
                    num_soc+=1
                if dict[i]['subject'] == 'language science':
                    num_lan+=1
                if dict[i]['grade'] in ['grade1', 'grade2', 'grade3', 'grade4', 'grade5', 'grade6']:
                    num_g16+=1
                if dict[i]['grade'] in ['grade7', 'grade8', 'grade9', 'grade10', 'grade11', 'grade12']:
                    num_g712+=1
            num+=1
    results = {
        'acc': float(correct)/num,
        'acc_nat': float(correct_nat)/num_nat,
        'acc_soc': float(correct_soc)/num_soc,
        'acc_lan': float(correct_lan)/num_lan,
        'acc_g16': float(correct_g16)/num_g16,
        'acc_g712': float(correct_g712)/num_g712,
    }
    print(results)
    wandb.log(results)

    return results