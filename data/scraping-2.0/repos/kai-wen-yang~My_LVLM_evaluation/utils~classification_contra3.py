import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Optional
from collections import defaultdict
import pdb
import torch
from .tools import has_word, remove_special_chars
import sys 
sys.path.append("..") 
from my_eval import openai_classnames
from models import get_image
import wandb

vicuna_prompt =\
    'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions.\\nUSER: {}\nASSISTANT:'


def evaluate_zero_shot_image_classification_contra3(
    model,
    dataset,
    model_name,
    dataset_name,
    time,
    batch_size=1,
    answer_path='answers',
    question='What is the object in the image?',
    max_new_tokens=16,
    per_class_acc=True,
    cot=None,
    cot_position='end',
    args=None,
):
    classnames = dataset.dataset.classnames
    predictions=[]
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})
    i = 0
    for t, batch in enumerate(tqdm(dataloader, desc="Running inference")):

        questions = []
        for i in range(len(batch['image_path'])):
            options_a = batch['options'][i][0]
            options_b = batch['options'][i][1]
            questions.append(vicuna_prompt.format(f'What is the difference between {options_a} and {options_b}? Answer it briefly.'))

        rationale = model.text_generate(questions, max_new_tokens=128)
        questions_all = []
        for i in range(len(batch['image_path'])):
            questions_all.append(rationale[i]+f'<Img><HereForImage></Img> Based on these information, is the object in the image a {options_a} or a {options_b}?')

        outputs = model.batch_generate(batch['image_path'], questions_all, max_new_tokens=max_new_tokens)

        j = 0
        for image_path, gt_answer, output, question, option, label, conf in zip(batch['image_path'], batch['gt_answers'],outputs, questions_all, batch['options'], batch['label'], batch['confidence']):
            if type(image_path) is not str:
                image_path = f'batch#{i} sample#{j}'
            answer_dict={'question': question, 'answer': output,
            'gt_answers': gt_answer, 'image_path': image_path, 'confidence':conf,
            'model_name': model_name, 'clip_prediction': option[0], 'label':label}

            predictions.append(answer_dict)
            j += 1

        if type(gt_answer) is not str:
            gt_answer = gt_answer[0]
        text_table = wandb.Table(columns=["answer", "label", "option", "confidence", "question"])
        text_table.add_data(output, gt_answer, ', '.join(batch['options'][i][:args.top_option]), str(conf), question)
        wandb.log({f'time{time}_batch{str(t)}/image.jpg': wandb.Image(get_image(image_path)),
                   f'time{time}_batch{str(t)}/table': text_table,
                   })
        i += 1

    answer_dir = os.path.join(answer_path, time)

    os.makedirs(answer_dir, exist_ok=True)
    answer_path = os.path.join(answer_dir, f"{dataset_name}.json")
    with open(answer_path, "w") as f:
        f.write(json.dumps(predictions, indent=4))
    correct = 0
    num = 0
    exact_match = 0
    clip_match = 0
    clip_unmatch_llm_match = 0
    clip_unmatch = 0
    clip_match_conf = 0
    clip_unmatch_conf = 0
    high_clip_low_llm = 0
    per_class_dict = defaultdict(lambda : defaultdict(int))
    with open(answer_path, 'r') as f:
        dict = json.load(f)
        for i in range(len(dict)):

            answer = dict[i]['answer']
            answer = remove_special_chars(answer).lower()
            gt_answers = dict[i]['gt_answers']
            if type(gt_answers) is str:
                cls_name = gt_answers
                gt_answers = [remove_special_chars(gt_answers).lower()]
            else:
                cls_name = gt_answers[0]
                gt_answers = [remove_special_chars(x).lower() for x in gt_answers]
            per_class_dict[cls_name]['total'] += 1
            if any([has_word(answer, x) for x in gt_answers]):
                per_class_dict[cls_name]['correct'] += 1
                correct+=1
            if any([answer == x for x in gt_answers]):
                exact_match += 1
            if dict[i]['clip_prediction'] == classnames[dict[i]['label']]:
                clip_match += 1
                clip_match_conf += dict[i]['confidence']
            else:
                clip_unmatch += 1
                clip_unmatch_conf += dict[i]['confidence']
                if any([has_word(answer, x) for x in gt_answers]):
                    clip_unmatch_llm_match+=1
            if (dict[i]['confidence']>0.25 and dict[i]['clip_prediction'] == classnames[dict[i]['label']]) or (dict[i]['confidence']<=0.25 and any([has_word(answer, x) for x in gt_answers])):
                high_clip_low_llm +=1

            num+=1
    acc_has_word = correct / num * 100
    acc_exact_match = exact_match / num * 100
    print(f'{dataset_name} of has_word: {acc_has_word:.2f}%')
    print(f'{dataset_name} of exact match: {acc_exact_match:.2f}%')
    print(f'{dataset_name} of clip match: {clip_match / num * 100:.2f}%')
    print(f'{dataset_name} of high clip low llm: {high_clip_low_llm / num * 100:.2f}%')
    print(f'{dataset_name} of clip unmatch: {clip_unmatch / num * 100:.2f}%')
    print(f'{dataset_name} of clip unmatch llm match: {clip_unmatch_llm_match / clip_unmatch * 100:.2f}%')
    print(f'{dataset_name} of clip match confidence: {clip_match_conf / clip_match:.2f}%')
    print(f'{dataset_name} of clip unmatch confidence: {clip_unmatch_conf / clip_unmatch:.2f}%')

    metrics = {
        'has_word': acc_has_word,
        'exact match': acc_exact_match,
    }
    if per_class_acc:
        num_classes = len(per_class_dict)
        acc_sum = 0.0
        for val in per_class_dict.values():
            acc_sum += val['correct'] / val['total']
        mean_per_class_acc = acc_sum / num_classes * 100
        metrics['mean_per_class_acc'] = mean_per_class_acc
        print(f'{dataset_name} of mean per-class: {mean_per_class_acc:.2f}%')
    return metrics
