"""
Automatic instruction generation with GPT-3
Automatically collect the generated instruction from GPT-3 and save as prompt files
"""

import os
import sys
import json
import pdb
import time
import openai
import argparse
from tqdm import tqdm
from typing import List
sys.path.append('./GPT-3/optimization')
sys.path.append('./GPT-3')

import tiktoken
tokenizer = tiktoken.encoding_for_model("text-davinci-003")
from Constants import NIV2_OUT_DOMAIN_TEST_TASKS_SET2
from utils.gpt_utils import gpt_inference, chatgpt_single_turn_inference
TEST_TASKS = NIV2_OUT_DOMAIN_TEST_TASKS_SET2

CHATGPT_SERIES = ["gpt-3.5-turbo", "gpt-4"]
GPT3_SERIES = ["text-davinci-003", "text-davinci-002", "code-davinci-002"]
META_PROMPT_LIST = [
    "niv2_demo_task_with_demo_1",
    "niv2_demo_task_with_demo_3",
    "niv2_demo_task_with_demo_5",
    "one-paragraph_instruction_with_demo",
    "one-sentence_instruction_with_demo",
    "step-by-step_instruction_with_demo",
    "use-example_instruction_with_demo"
]


def main():
    parser = argparse.ArgumentParser()
    # GPT-3 API arguments
    parser.add_argument('-model', type=str, default='text-davinci-003',
                        help='Which GPT-3 model to use')
    parser.add_argument('-max_tokens', type=int, default=512,
                        help='Max number of output tokens of GPT-3')
    parser.add_argument('-temperature', type=float, default=1.0,
                        help='Temperature parameter of GPT-3. 0: greedy decoding. 1: sampling based on probability')
    parser.add_argument('-top_p', type=float, default=1.0,
                        help='Sampling probability threshold in nucleus sampling')
    parser.add_argument('-best_of', type=int, default=5,
                        help='Maximum number of sampled outputs')
    parser.add_argument('-num_return', type=int, default=3,
                        help='How many output sequences to return. Note that this argument also applies to ChatGPT '
                             'if using ChatGPT to generate multiple instructions')
    parser.add_argument('-logprobs', type=int, default=None,
                        help='The number of tokens to return probability')
    parser.add_argument('-timeout', type=int, default=30,
                        help='Wait time for the response from OpenAI API')
    parser.add_argument('-sleep', type=int, default=2,
                        help='How long to sleep if rate limit is reached')
    parser.add_argument('-api_key', type=str, required=True, help='OPENAI API key')

    # Data arguments
    # training_data/out-domain/flan
    parser.add_argument('-data_dir', type=str, default='data/niv2_english',
                        help='Data directory')
    parser.add_argument('-meta_prompt_file_name', type=str, default="meta_prompt.txt",
                        help='Meta prompt file name in each downstream task directory')
    # text003_prompt_5.jsonl
    parser.add_argument('-output_file_name', type=str, default="text003_prompt_3.json",
                        help='Output file name to save the generated instructions')

    args = parser.parse_args()

    task_names = TEST_TASKS
    start_time = time.time()
    print(f'Start: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
    total_read_tokens = 0
    total_generate_tokens = 0

    for i, task in enumerate(task_names):
        print(f"{'-'*15}{task}({i + 1}/{len(task_names)}){'-'*15}")

        for meta_prompt_name in tqdm(META_PROMPT_LIST, desc='Iterating candidate meta prompts'):
            meta_prompt_path = os.path.join(args.data_dir, task, meta_prompt_name, args.meta_prompt_file_name)
            with open(meta_prompt_path, 'r', encoding='utf8') as fin:
                meta_prompt = fin.read().strip()

            # output_path: jsonl file for saving GPT-3 prompt
            # output_text_path: txt file for saving the generated instruction only
            output_path = os.path.join(args.data_dir, task, meta_prompt_name, args.output_file_name)
            output_text_path = os.path.join(args.data_dir, task, meta_prompt_name,
                                            args.output_file_name[:-5] + ".txt")
            if os.path.exists(output_path):
                print(f'Output path {output_path} already exists! Skip to next task...')
                continue

            # Use meta prompt to generate instructions
            # For ChatGPT, use OpenAI Chat API
            if args.model in CHATGPT_SERIES:
                openai.api_key = args.api_key
                instruction_list = chatgpt_single_turn_inference(
                    inputs_with_prompts=meta_prompt,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    num_return=args.num_return,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stop=None,
                    timeout=args.timeout,
                    sleep=args.sleep
                )

            # For GPT-3 models, call OpenAI API
            elif args.model in GPT3_SERIES:

                openai.api_key = args.api_key
                instruction_list, logprob_dict = gpt_inference(
                    inputs_with_prompts=meta_prompt,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    num_return=args.num_return,
                    best_of=args.best_of,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    logprobs=args.logprobs,
                    stop=None,
                    timeout=args.timeout,
                    sleep=args.sleep
                )

            else:
                raise ValueError(f"Invalid model name {args.model}!")

            assert len(instruction_list) == args.num_return

            with open(output_text_path, 'w', encoding='utf8') as fout_text:
                for instruction in instruction_list:
                    fout_text.write(instruction.strip())
                    fout_text.write('\n\n\n' + '-' * 50 + '\n\n\n')
                    total_generate_tokens += len(tokenizer.encode(instruction.strip()))

            total_read_tokens += len(tokenizer.encode(meta_prompt.strip()))

            json.dump(instruction_list, open(output_path, 'w', encoding='utf8'), ensure_ascii=False, indent=4)

    print(f'End: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
    print(f"Total time: {time.time() - start_time:.4f}s")
    print(f'Total read tokens: {total_read_tokens}')
    print(f'Total generate tokens: {total_generate_tokens}')


if __name__ == "__main__":
    main()
