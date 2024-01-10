import argparse
import json
import os
import time

import openai
from tqdm import tqdm
import ray

import shortuuid
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
import torch

from fastchat.model import get_conversation_template


@ray.remote(num_gpus=1)
@torch.inference_mode()
def get_reviews(model_path, model_id, question_jsons, answer1_jsons, answer2_jsons, reviewer_jsons, prompt_jsons):
    model_path = os.path.expanduser(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, low_cpu_mem_usage=True, torch_dtype=torch.float16
    ).cuda()
    
    review_jsons = []
    for i, line in enumerate(tqdm(question_jsons)):
        assert (
            answer1_jsons[i]["question_id"]
            == question_jsons[i]["question_id"]
            == answer2_jsons[i]["question_id"]
        )

        ques = question_jsons[i]["text"]
        cat = question_jsons[i]["category"]
        ans1 = answer1_jsons[i]["text"]
        ans2 = answer2_jsons[i]["text"]
        sys_prompt, prompt, reviewer_id = gen_prompt(
            reviewer_jsons, prompt_jsons, cat, ques, ans1, ans2
        )

        # handles.append(get_eval.remote(args.model_path, args.model_id, sys_prompt, prompt, args.max_tokens))
        # reviews.append(get_eval(args.model_path, args.model_id, sys_prompt, prompt, args.max_tokens))
        conv = get_conversation_template(model_id)
        conv.system = sys_prompt
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer([prompt]).input_ids
        output_ids = model.generate(
            torch.as_tensor(input_ids).cuda(),
            # do_sample=True,
            temperature=0.7,
            max_new_tokens=1024,
        )
        output_ids = output_ids[0][len(input_ids[0]) :]
        outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        # import ipdb; ipdb.set_trace()
        # scores = parse_three_class_score(outputs)
        if 'threeclass' in args.prompt_file:
            scores = parse_three_class_score(outputs)
        elif 'score_to_three' in args.prompt_file:
            scores = parse_score_to_three_score(outputs)            
        else:
            scores = parse_score(outputs)

        review_id = shortuuid.uuid()
        review_jsons.append(
            {
                "review_id": review_id,
                "question_id": question_jsons[i]["question_id"],
                "answer1_id": answer1_jsons[i]["answer_id"],
                "answer2_id": answer2_jsons[i]["answer_id"],
                "reviewer_id": reviewer_id,
                "metadata": {},
                "text": outputs,
                "score": scores
            }
        )

    return review_jsons

def parse_three_class_score(review):
    try:
        score = int(review.strip().split("\n")[-1].strip())
        return score
    except Exception as e:
        logger.error(
            f"{e}\nContent: {review}\n" "You must manually fix the score pair."
        )
        return -1

def parse_score_to_three_score(review):
    try:
        score_pair = review.split("\n")[0]
        score_pair = score_pair.replace(",", " ")
        sp = score_pair.split(" ")
        if len(sp) == 2:
            if abs(float(sp[0])-float(sp[1])) < 1e-2:
                return 3
            elif float(sp[0]) > float(sp[1]):
                return 1
            elif float(sp[0]) < float(sp[1]):
                return 2

        else:
            raise Exception("Invalid score pair.")
    except Exception as e:
        logger.error(
            f"{e}\nContent: {review}\n" "You must manually fix the score pair."
        )
        return -1

def parse_score(review):
    try:
        score_pair = review.split("\n")[0]
        score_pair = score_pair.replace(",", " ")
        sp = score_pair.split(" ")
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            raise Exception("Invalid score pair.")
    except Exception as e:
        logger.error(
            f"{e}\nContent: {review}\n" "You must manually fix the score pair."
        )
        return [-1, -1]


def gen_prompt(reviewer_jsons, prompt_jsons, cat, ques, ans1, ans2):
    # Default to general category (index=0)
    reviewer_idx = 0
    for idx, reviewer in enumerate(reviewer_jsons):
        if reviewer["category"] == cat:
            reviewer_idx = idx
            break
    prompt_id = reviewer_jsons[reviewer_idx]["prompt_id"]
    prompt_json = prompt_jsons[prompt_id - 1]
    assert prompt_json["prompt_id"] == prompt_id

    sys_prompt = prompt_json["system_prompt"]
    prompt_template = prompt_json["prompt_template"]
    defaults = prompt_json["defaults"]
    prompt = prompt_template.format(
        question=ques, answer_1=ans1, answer_2=ans2, **defaults
    )

    return sys_prompt, prompt, reviewer_idx + 1


def get_json_list(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, "r") as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
        return json_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model-based QA evaluation.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("-q", "--question-file")
    parser.add_argument("-a", "--answer-file-list", nargs="+", default=[])
    parser.add_argument("-p", "--prompt-file")
    parser.add_argument("-r", "--reviewer-file")
    # parser.add_argument("-m", "--reviewer-model")
    parser.add_argument("-o", "--output-review-file")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="maximum number of tokens produced in the output",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.output_review_file):
        dest = args.output_review_file
    else:
        threeclass_suff = "_threeclass" if 'threeclass' in args.prompt_file else ""
        # import ipdb; ipdb.set_trace()
        dest = os.path.join(
            args.output_review_file,
            '-vs-'.join([elt.split('/')[-1].replace('.jsonl', '') for elt in args.answer_file_list]) + f'-{args.model_id}-reviewer{threeclass_suff}' + '.jsonl'
        )

    ray.init()

    question_jsons = get_json_list(args.question_file)
    answer1_jsons = get_json_list(args.answer_file_list[0])
    answer2_jsons = get_json_list(args.answer_file_list[1])
    reviewer_jsons = get_json_list(args.reviewer_file)
    prompt_jsons = get_json_list(args.prompt_file)

    # check if # of questions, answers are the same
    assert len(question_jsons) == len(answer1_jsons) == len(answer2_jsons)

    chunk_size = len(question_jsons) // args.num_gpus
    review_handles = []
    for i in range(0, len(question_jsons), chunk_size):
        review_handles.append(
            get_reviews.remote(
                args.model_path, args.model_id, question_jsons[i : i + chunk_size], answer1_jsons[i: i+chunk_size], answer2_jsons[i: i+chunk_size], reviewer_jsons, prompt_jsons
            )
    )

    review_jsons = []
    for review_handle in review_handles:
        review_jsons.extend(ray.get(review_handle))

    with open(dest, 'w') as output_review_file:
        for line in review_jsons:
            output_review_file.write(json.dumps(line) + "\n")
