import json
import os
import random
from argparse import ArgumentParser
from pathlib import Path

import openai

from main import (Question, build_prompt)

openai.organization = os.environ["OPENAI_ORGANIZATION"]
openai.api_key = os.environ["OPENAI_API_KEY"]

SKIP_REASONS = ("answer key is wrong", )


def get_keys(data: dict[str, dict], max_prompts: int) -> list[str]:
    """Get keys to sample for selecting which questions to repeatedly prompt
    GPT to answer. You can set which reasons the model missed the question to
    ignore. Currently, we only ignore questions where the answer key was wrong
    """
    right_keys = [k for k, v in data.items() if v['correct']]
    wrong_keys = [
        k for k, v in data.items()
        if not v['correct'] and v["classification"] not in SKIP_REASONS
    ]
    random.shuffle(right_keys)
    random.shuffle(wrong_keys)

    n_right = max_prompts // 2
    n_wrong = max_prompts - n_right

    right_key_sample = [key for key in right_keys[:n_right]]
    wrong_key_sample = [key for key in right_keys[:n_wrong]]

    return [*right_key_sample, *wrong_key_sample]


def process_record(data: dict[str, dict], key: str, model: str, n_reps: int,
                   output_data: dict[str, dict], output_file: Path) -> None:
    """Repeatedly prompt GPT for a single question `n_reps` times, print results
    to the console, then update the output record
    """
    key_data = data[key]
    output = []
    answer_cnts = {}
    originally_correct = key_data['correct']
    n_right = 0
    question = Question(
        key_data['question_type'],
        key_data['story_type'],
        key_data['actions'].split('\n'),
        key_data['question'],
        key_data['answer'],
    )
    prompt = build_prompt(question)

    for _ in range(n_reps):
        response = openai.ChatCompletion.create(
            model=model,
            temperature=0,
            messages=[
                {
                    "role":
                    "system",
                    "content":
                    "You are a highly analytical, detail-oriented assistant."
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ])

        gpt_response = response['choices'][0]['message']['content']
        gpt_answer = gpt_response.split("<answer>")[1].split(
            "</answer>")[0].lower()
        answer = question.answer
        correct = gpt_answer == answer
        n_right += int(correct)

        output.append({
            'answer': gpt_answer,
            'response': gpt_response,
            'correct': correct,
        })
        answer_cnts[gpt_answer] = answer_cnts.get(gpt_answer, 0) + 1
        print(gpt_answer, gpt_answer == answer)

    output_data[key] = {
        'originally_correct': originally_correct,
        'answers': output,
        'n_answers': len(answer_cnts),
        'pct_right': n_right / n_reps,
        'model': model,
    }

    json.dump(output_data, output_file.open('w'))
    print(answer_cnts)


def main(
    input_file: Path,
    output_file: Path,
    n_questions: int,
    n_reps: int,
    model: str,
) -> None:
    """Run core loop that repeatedly prompts GPT
    """
    data = json.load(input_file.open('r'))
    try:
        out_data = json.load(output_file.open('r'))
    except FileNotFoundError:
        out_data = {}

    keys = get_keys(data, n_questions)
    for k in keys:
        if k in out_data:
            continue

        process_record(data, k, model, n_reps, out_data, output_file)


if __name__ == "__main__":
    parser = ArgumentParser(
        description='This script runs repeatedly runs a prompt against a model'
    )

    parser.add_argument(
        '-f',
        '--file',
        type=Path,
        default='./data/classified_results.json',
        required=False,
        help='Path to the input file. Default is ./data/classified_results.json'
    )
    parser.add_argument(
        '-o',
        '--output_file',
        type=Path,
        default='./data/repeat_query_output.json',
        required=False,
        help=
        'Path to the output file. Default is ./data/repeat_query_output.json')
    parser.add_argument(
        '-q',
        '--n_questions',
        type=int,
        default=20,
        required=False,
        help=
        'The number of questions to repeatedly call model against. Default is 20'
    )
    parser.add_argument(
        '-r',
        '--n_reps',
        type=int,
        default=10,
        required=False,
        help=
        'The number of times to call the model on for each prompt. Default is 10'
    )
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        default='gpt-4',
        required=False,
        help=
        'The name of the OpenAI model to be used for question answering. Default is gpt-4.'
    )

    args = parser.parse_args()
    main(args.file, args.output_file, args.n_questions, args.n_reps,
         args.model)
