# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import argparse
import copy
import json
import os
import random
import sys
import openai
import dataclasses
import logging
import tenacity
import tiktoken
from tqdm import tqdm
from typing import Optional, Sequence, Union, List
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

from prompt_templates import ClassificationValidationPrompt
from chat_completion import openai_chat_completion


def default_stop() -> List[str]:
    return ["None.", "None", "none.", "none"]

@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 2048
    temperature: float = 0.2
    top_p: float = 0.0
    n: int = 1
    stream: bool = False
    # stop: Optional[List[str]] = dataclasses.field(default_factory=default_stop)
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

def save_intermediate_results(all_items, args, message):
    file_name = os.path.basename(args.save_file)
    file_name = file_name.rsplit(".", 1)[0] + f".{message}.json"
    terminate_save_path = os.path.join(args.path, "terminated_results")
    os.makedirs(terminate_save_path, exist_ok=True)
    with open(os.path.join(terminate_save_path, file_name), "w") as f:
        json.dump(all_items, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, 
                        default=None, help="not recommended; better to set env varaible.")
    parser.add_argument("--api_name", type=str, default="gpt-3.5-turbo", help="the name of the api model.")
    parser.add_argument("--path", type=str, 
                        default='./data/SuperNI_v4', help='source file & target save path.')
    parser.add_argument("--data_file", type=str,
                        default='original_collection.json')
    parser.add_argument("--save_file", type=str,
                        default='classified_superni_cls_instructions.json')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="overwrite the save file if it exists.")
    parser.add_argument("--instance_num", type=int, default=None, help="number of instances (input) to annotate.")
    parser.add_argument("--demo_instructions", type=str, default="./data/only_CLS/demon_instructions.json",
                        help="path to the demonstration instructions (cls tasks).")

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    openai.api_key = os.getenv("OPENAI_API_KEY") if args.api_key is None else args.api_key
    args.data_file = os.path.join(args.path, args.data_file)
    args.save_file = os.path.join(args.path, args.save_file)
    random.seed(args.seed)
    
    if os.path.exists(args.save_file) and not args.overwrite:
        raise ValueError("Save file {} already exists, set --overwrite to overwrite it.".format(args.save_file))

    template = ClassificationValidationPrompt()
    decoding_args = OpenAIDecodingArguments()

    # read the input files
    if os.path.exists(args.data_file):
        with open(args.data_file, "r") as f:
            instances = json.load(f)
    else:
        raise ValueError("Input file {} does not exist.".format(args.data_file))
    
    # read the demonstration instructions
    with open(args.demo_instructions, "r") as f:
        demo_instances = json.load(f)
        if isinstance(demo_instances[0], list):
            demo_instances = [item[0] for item in demo_instances]
        assert isinstance(demo_instances[0], str)

    all_items, skip_num, complete_num = [], 0, 0
    cls_instruction_cnt = []
    # randomly sample subset of instances (when testing)
    instances = random.sample(instances, min(args.instance_num, len(instances))) if args.instance_num is not None else instances
    
    try:
        for i, instance in tqdm(enumerate(instances), total=len(instances)):
            id = instance["id"]
            input = instance["input"]
            official_instruction = instance["instruction"]
            official_output = instance["output"][0]
            appliable_instructions, outputs = [], []
            for demo_instruction in demo_instances:
                if demo_instruction == official_instruction:
                    # if the demo instruction is the official instruction, skip
                    appliable_instructions.append(demo_instruction)
                    outputs.append(official_output)
                    continue
                query = {"id": id, "input": input, "instruction": demo_instruction}
                content, cost = openai_chat_completion(query, template, decoding_args, model_name=args.api_name)
                if content is None:
                    skip_num += 1
                    continue
                complete_num += 1
                if content == "yes":
                    appliable_instructions.append(demo_instruction)
                else:
                    continue
            new_item = {"id": id, "input": input, "hint": "", "cost": cost, "example_1": "", "example_2": "", "example_3": "", 
                        "instructions": appliable_instructions, "outputs": outputs}
            all_items.append(new_item)
            cls_instruction_cnt.append(len(appliable_instructions))
    except KeyboardInterrupt as e:
        # save the intermediate results
        print("==> Error: {}".format(e))
        print("\nUser terminated the program\n")
        save_intermediate_results(all_items, args, "KeyboardInterrupt")
        sys.exit(0)  # Exit the program gracefully
    # except openai.error.RateLimitError as e:
    except tenacity.RetryError as e:
        print("==> Error: {}".format(e))
        print("\nOpenAI API rate limit reached. Please increase the waiting/retry times in the tenacity decorator.\n")
        save_intermediate_results(all_items, args, "RateLimitError")
        sys.exit(0)  # Exit the program gracefully

    # write the output files
    # save_file = args.data_file.replace(".json", "_attributes.json")
    save_file = args.save_file
    with open(save_file, "w") as f:
        json.dump(all_items, f, indent=2)

    print("==> saved to {}".format(save_file))
    print("==> skip: {} ; complete: {}".format(skip_num, complete_num))
    print("==> total number of applicable instructions: {}".format(sum(cls_instruction_cnt)))
    print("==> average number of applicable instructions per input: {:.2f}".format(sum(cls_instruction_cnt) / len(cls_instruction_cnt)))
    # save above screen print to a file
    file_name = args.save_file.split("/")[-1].split(".")[0]
    screen_save_path = os.path.join(args.path, "screen_print")
    os.makedirs(screen_save_path, exist_ok=True)
    with open(os.path.join(screen_save_path, file_name + ".txt"), "w") as f:
        f.write("==> saved to {}\n".format(save_file))
        f.write("==> skip: {} ; complete: {}".format(skip_num, complete_num))
        f.write("==> total number of applicable instructions: {}".format(sum(cls_instruction_cnt)))
        f.write("==> average number of applicable instructions per input: {:.2f}".format(sum(cls_instruction_cnt) / len(cls_instruction_cnt)))
        

if __name__ == "__main__":
    main()
