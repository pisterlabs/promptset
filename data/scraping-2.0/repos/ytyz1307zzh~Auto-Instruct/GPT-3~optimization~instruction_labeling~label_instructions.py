"""
Label all the <instruction, example> pairs using GPT-3
If the instruction can lead to a correct prediction on the example, then it is labeled as 1
"""

import os
import pdb
import sys
import platform
if platform.platform().startswith("Windows"):
    script_path = os.path.abspath(__file__)
    sys.path.append('\\'.join(script_path.split('\\')[:-3]))
else:
    script_path = os.path.abspath(__file__)
    sys.path.append('/'.join(script_path.split('/')[:-3]))

import re
import json
import math
import time
import random
import shutil
import openai
import argparse
import numpy as np
from tqdm import tqdm
from typing import List
from rouge_score import rouge_scorer
from utils.gpt_utils import gpt_inference, chatgpt_single_turn_inference

EVAL_METRICS = [
    "exact_match",
    "norm_exact_match",
    "prefix_match",
    "norm_prefix_match",
    "rouge_L",
    "relative_rouge_L"
]

random.seed(46556)
RougeScorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

GPT3_SERIES = [
    "text-davinci-003",
    "text-davinci-002",
    "code-davinci-002",
]

CHATGPT_SERIES = [
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4-0314"
]


def read_jsonl_as_list(path: str):
    assert path.endswith('.jsonl')
    with open(path, 'r', encoding='utf8') as fin:
        result = []
        for line in fin:
            data = json.loads(line.strip())
            result.append(data)
    # print(f'Read {len(result)} data from {path}')
    return result


def save_list_as_jsonl(path: str, data: List):
    assert path.endswith('.jsonl')
    with open(path, 'w', encoding='utf8') as fout:
        for instance in data:
            fout.write(json.dumps(instance))
            fout.write('\n')
    # print(f'Saved {len(data)} data to {path}')


def read_json(path: str):
    assert path.endswith('.json')
    result = json.load(open(path, 'r', encoding='utf8'))
    # print(f'Read {len(result)} data from {path}')
    return result


def save_json(path: str, data):
    assert path.endswith(".json")
    json.dump(data, open(path, 'w', encoding='utf8'), ensure_ascii=False, indent=4)
    # print(f'Saved {len(data)} data to {path}')


def read_txt(path: str):
    with open(path, 'r', encoding='utf8') as fin:
        text = fin.read().strip()
    return text


def normalize_answer(s: str):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    return white_space_fix(remove_articles(s))


def format_demo(all_demos, demo_template_info):
    single_example_template = demo_template_info["template"]
    delimiter = demo_template_info["delimiter"]
    all_demo_strings = []

    for demo_example in all_demos:
        input_text = demo_example["input"]
        output_text = demo_example["target"]
        demo_string = single_example_template.replace("{{input}}", input_text)
        demo_string = demo_string.replace("{{target}}", output_text)
        all_demo_strings.append(demo_string)

    formatted_demo = delimiter.join(all_demo_strings)
    return formatted_demo


def answer_match(prediction: str, answer: str, metric: str):
    prediction = prediction.lower().strip()
    answer = answer.lower().strip()

    # If prediction or the answer is empty, directy return False
    if len(prediction) == 0 or len(answer) == 0:
        return False

    # Remove the period at the end of the prediction and answer, if exists
    if prediction[-1] == '.':
        prediction = prediction[:-1]
    if answer[-1] == '.':
        answer = answer[:-1]

    if metric == 'exact_match':
        accuracy = (prediction == answer)
    elif metric == "norm_exact_match":
        accuracy = (normalize_answer(prediction) == normalize_answer(answer))
    elif metric == 'prefix_match':
        accuracy = (prediction[:len(answer)] == answer)
    elif metric == 'norm_prefix_match':
        prediction = normalize_answer(prediction)
        answer = normalize_answer(answer)
        accuracy = (prediction[:len(answer)] == answer)
    elif metric == 'rouge_L':
        rouge_score = RougeScorer.score(target=answer, prediction=prediction)['rougeL'].fmeasure
        accuracy = (rouge_score >= 0.5)
    else:
        raise ValueError(f'Unknown metric: {metric}!')

    return accuracy


def sample_on_group(all_instructions: List[str], instruction_groups: List[int], num_samples: int):
    """
    Sample instructions based on the groups. Only at most one instruction will be sampled from a certain group.
    :param all_instructions: all instructions in a list
    :param instruction_groups: a list of numbers, each indicates how many instructions are in this group
    :param num_samples: the number of instructions to be sampled
    :return: a list of sampled instruction INDICES
    """
    assert sum(instruction_groups) == len(all_instructions)
    assert num_samples <= len(instruction_groups)
    all_instruction_indices = list(range(len(all_instructions)))

    grouped_instruction_indices = []
    for group_size in instruction_groups:
        grouped_instruction_indices.append(all_instruction_indices[:group_size])
        all_instruction_indices = all_instruction_indices[group_size:]

    sampled_instruction_index_each_group = [random.choice(group) for group in grouped_instruction_indices]
    final_sampled_instruction_indices = random.sample(sampled_instruction_index_each_group, num_samples)

    return final_sampled_instruction_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', type=str, required=True,
                        help='Task name, use "_" instead of whitespaces to connect words')

    # File related arguments
    parser.add_argument('-instruction_file', type=str, required=True,
                        help='File containing instructions for the task. '
                             'Should be a JSON file which contains a list of instruction strings.')
    parser.add_argument('-data_file', type=str, required=True,
                        help='File containing the training data. '
                             'Should be a JSONL file, where each line is a dict with fields '
                             '"id", "input" and "target".')
    parser.add_argument('-demo_file', type=str, required=True,
                        help='File containing the demonstration examples. '
                             'Should be a JSONL file, where each line is a dict with fields '
                             '"id", "input" and "target". Treat all tasks in the QA format.')
    parser.add_argument('-gpt_inference_prompt_file', type=str, required=True,
                        help='File containing the prompt template for GPT-3 to solve the task. '
                             'Should be a text file which includes the demonstration examples, '
                             'as well as placeholders for the instruction and the test example')
    parser.add_argument('-demo_template_file', type=str, required=True,
                        help='File containing the template of the demo examples. '
                             'Should be a JSON file which includes a template '
                             'with placeholders for the input and output, and also the delimiter')
    parser.add_argument('-instruction_selection_prompt_file', type=str, required=True,
                        help='File containing the prompt template for further instruction selection')
    parser.add_argument('-temp_save_file', type=str, default='temp_save.jsonl',
                        help='File to save intermediate results')
    parser.add_argument('-instruction_group_file', type=str, default=None,
                        help='If not None, then sample instructions based on the groups. '
                             'Instructions will not be sampled from the same group. '
                             'args.sample_instructions must not be None.')
    parser.add_argument('-output_file', type=str, required=True,
                        help='The file to save instruction labeling results')

    # OpenAI API related arguments
    parser.add_argument('-model', type=str, default='text-davinci-003',
                        help='Which GPT-3 model to use')
    parser.add_argument('-batch_size', type=int, default=20,
                        help='How many prompts to send together to GPT3')
    parser.add_argument('-sample_instructions', type=int, default=None,
                        help='If not None, randomly sample a subset of instructions for each example')
    parser.add_argument('-sample_data', type=int, default=None,
                        help='If not None, only use the first sample_data examples')
    parser.add_argument('-max_tokens', type=int, default=128,
                        help='The max number of tokens for GPT generation')
    parser.add_argument('-timeout', type=int, default=20,
                        help='If API respond time exceeds this value, then call it again')
    parser.add_argument('-sleep', type=int, default=1,
                        help='How long to sleep if rate limit is reached')
    parser.add_argument('-api_key', type=str, required=True, help='OpenAI API key')

    # Evaluation related arguments
    parser.add_argument('-eval_metric', type=str, choices=EVAL_METRICS,
                        default='exact_match',
                        help='How to evaluate GPT output. '
                             'exact_match: the GPT output must exactly match the answer. '
                             'prefix_match: the answer must be a prefix of the GPT output. '
                             'rouge_L: calculate ROUGE score.')

    # async related arguments
    parser.add_argument('-use_async', action='store_true', default=False, help='Whether to use async API call')
    parser.add_argument('-num_async_workers', default=None, type=int,
                        help='If use async API, how many async jobs to create in parallel.')

    args = parser.parse_args()

    if args.instruction_group_file is not None:
        assert args.sample_instructions is not None

    # File for temporary saving intermediate results, in case the program crashed halfway through
    temp_save_file = args.temp_save_file
    previous_data = None
    if os.path.exists(temp_save_file):
        previous_data = read_jsonl_as_list(temp_save_file)
        assert len(previous_data) == 1
        previous_data = previous_data[0]
        previous_instruction_indices = previous_data["instruction_indices"]
        previous_outputs = previous_data["outputs"]
        previous_answers = previous_data["answers"]
    ftemp = open(temp_save_file, 'a', encoding='utf8')

    batch_size = args.batch_size
    openai.api_key = args.api_key
    assert not os.path.exists(args.output_file), "The output file already exists!"
    with open(args.output_file, 'a', encoding='utf8') as fout:
        pass  # verify that the directory exists
    os.remove(args.output_file)

    all_instructions = read_json(args.instruction_file)
    all_data = read_jsonl_as_list(args.data_file)
    all_demos = read_jsonl_as_list(args.demo_file)
    demo_template_info = read_json(args.demo_template_file)
    prompt_template = read_txt(args.gpt_inference_prompt_file)
    instruction_selection_prompt = read_txt(args.instruction_selection_prompt_file)
    print(f'Total examples to label: {len(all_data)}')

    if args.sample_instructions is not None:
        num_instructions_per_example = args.sample_instructions
    else:
        num_instructions_per_example = len(all_instructions)

    num_previous_examples = 0
    if previous_data is not None:
        assert len(previous_outputs) % num_instructions_per_example == 0
        num_previous_examples = len(previous_outputs) // num_instructions_per_example
        remain_data = all_data[num_previous_examples:]
        print(f'Found previous saved results: {num_previous_examples}')
        print(f'Besides previous saved results, there are {len(remain_data)} in total.')
    else:
        remain_data = all_data

    if args.sample_data is not None:
        request_examples = min(args.sample_data, len(all_data))
    else:
        request_examples = len(all_data)
    remain_examples = request_examples - num_previous_examples
    print(f'In this run, there are {remain_examples} remaining.')

    # The ground-truth answer for each input
    all_answers = previous_answers if previous_data is not None else []
    # Which instruction corresponds to each input
    all_instruction_indices = previous_instruction_indices if previous_data is not None else []
    # all input prompts for all examples (before putting into batches)
    all_input_prompts = []
    remain_data = remain_data[:remain_examples]
    num_batches = math.ceil((len(remain_data) * num_instructions_per_example) / batch_size)

    # We combine all <exmaple, instruction> pairs together to batch them, in order to save time
    for data_example in remain_data:
        input_text = data_example["input"]
        output_text = data_example["target"]

        prompt = prompt_template.replace("{{input}}", input_text)
        demo = format_demo(all_demos, demo_template_info)
        prompt = prompt.replace("{{demo}}", demo)

        if args.sample_instructions is not None:

            # If instructions are grouped, then they are sampled based on this grouping
            if args.instruction_group_file is not None:
                instruction_groups = read_json(args.instruction_group_file)
                random_instruction_indices = sample_on_group(all_instructions, instruction_groups, args.sample_instructions)
            # Otherwise, randomly sample from all instructions
            else:
                random_instruction_indices = random.sample(list(range(len(all_instructions))), args.sample_instructions)

            random_instructions = [all_instructions[idx] for idx in random_instruction_indices]

        else:
            random_instruction_indices = [idx for idx in range(len(all_instructions))]
            random_instructions = all_instructions
        all_instruction_indices.extend(random_instruction_indices)

        for instruction in random_instructions:
            prompt_with_instruction = prompt.replace("{{instruction}}", instruction.strip())
            all_input_prompts.append(prompt_with_instruction.strip())
            all_answers.append(output_text)

    # The prediction of GPT-3 for each input
    all_outputs = previous_outputs if previous_data is not None else []
    for batch_idx in tqdm(range(num_batches), desc="Calling API"):
        batch_prompts = all_input_prompts[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        batch_outputs = []

        if args.model in GPT3_SERIES:
            batch_outputs, batch_logprob_dict = gpt_inference(
                inputs_with_prompts=batch_prompts,
                model=args.model,
                max_tokens=args.max_tokens,
                num_return=1,
                best_of=1,
                temperature=0.0,
                top_p=0.0,
                logprobs=None,
                echo=False,
                stop=None,
                timeout=args.timeout,
                sleep=args.sleep,
                use_async=args.use_async,
                num_async_workers=args.num_async_workers if args.use_async else None
            )

        elif args.model in CHATGPT_SERIES:
            if args.use_async:
                batch_outputs = chatgpt_single_turn_inference(
                    inputs_with_prompts=batch_prompts,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    num_return=1,
                    temperature=0.0,
                    top_p=0.0,
                    stop='\n\n',
                    timeout=args.timeout,
                    sleep=args.sleep,
                    use_async=args.use_async,
                    num_async_workers=args.num_async_workers
                )
            else:
                batch_outputs = []
                for single_prompt in batch_prompts:
                    single_output = chatgpt_single_turn_inference(
                        inputs_with_prompts=[single_prompt],
                        model=args.model,
                        max_tokens=args.max_tokens,
                        num_return=1,
                        temperature=0.0,
                        top_p=0.0,
                        stop='\n\n',
                        timeout=args.timeout,
                        sleep=args.sleep
                    )
                    batch_outputs.extend(single_output)

                    # sleep after each call of GPT-4 to reduce limit error
                    if args.model == "gpt-4":
                        time.sleep(1)

        else:
            raise ValueError(f"Invalid model name: {args.model}!")

        if not (isinstance(batch_outputs, list) and len(batch_outputs) == len(batch_prompts)):
            raise ValueError(f"Invalid output from GPT-3: {batch_outputs}")
        all_outputs.extend(batch_outputs)

        if len(all_outputs) % num_instructions_per_example == 0:
            # Create a copy first, in case the program crashes when writing the file
            shutil.copy(temp_save_file, temp_save_file + ".bak")
            ftemp.truncate(0)
            ftemp.write(json.dumps({
                "instruction_indices": all_instruction_indices[:len(all_outputs)],
                "outputs": all_outputs,
                "answers": all_answers[:len(all_outputs)]
            }))
            ftemp.flush()
            os.remove(temp_save_file + ".bak")

    # For each instruction, if the prediction matches with the answer, then it is a good instruction
    # Although we did this in the code, this label is not used in actual model training
    all_labels = []
    for output, answer in zip(all_outputs, all_answers):
        if answer_match(output, answer, metric=args.eval_metric):
            all_labels.append(1)
        else:
            all_labels.append(0)

    example_idx, output_idx = 0, 0
    all_labeled_data = []  # The final instruction labeling results
    assert len(all_labels) == request_examples * num_instructions_per_example

    # In all labels, find the corresponding ones for each example
    while example_idx < request_examples:
        data_example = all_data[example_idx]
        try:
            id_ = data_example["id"]
        except KeyError:
            id_ = example_idx
        input_text = data_example["input"]
        output_text = data_example["target"]

        instruction_indices_for_this_example = all_instruction_indices[output_idx: output_idx + num_instructions_per_example]
        labels_for_this_example = all_labels[output_idx: output_idx + num_instructions_per_example]
        outputs_for_this_example = all_outputs[output_idx: output_idx + num_instructions_per_example]
        output_idx += num_instructions_per_example

        rouge_for_this_example = [
            RougeScorer.score(target=output_text.strip(), prediction=prediction.strip())['rougeL'].fmeasure
            for prediction in outputs_for_this_example
        ]

        labeled_example = {
            "id": id_,
            "input": input_text,
            "target": output_text,
            "instruction_idx": instruction_indices_for_this_example,
            "instruction_labels": labels_for_this_example,
            "instruction_outputs": outputs_for_this_example,
            "instruction_rouge": rouge_for_this_example,
        }
        all_labeled_data.append(labeled_example)

        example_idx += 1

    task_name = args.task.replace("_", " ")
    task_instruction_selection_prompt = instruction_selection_prompt.replace("{{task}}", task_name)
    results = [
        {"prompt": task_instruction_selection_prompt},
        {"instructions": all_instructions},
        {"demos": format_demo(all_demos, demo_template_info)},
    ] + all_labeled_data

    save_list_as_jsonl(args.output_file, results)
    print(f'Saved {len(all_labeled_data)} data to {args.output_file}')
    ftemp.close()
    os.remove(temp_save_file)


if __name__ == "__main__":
    main()
