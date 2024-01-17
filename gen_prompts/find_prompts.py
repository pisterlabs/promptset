import json
import os
import shutil
from argparse import ArgumentParser
from glob import glob
from itertools import islice
from multiprocessing import Pool
from parsers import (
    PromptDetector,
    used_langchain_tool,
    used_chat_function,
    used_in_langchain_llm_call,
    used_in_openai_call,
    used_prompt_or_template_name,
    used_langchain_tool_class,
    new_line_in_string,
    find_from_file,
    all_strings,
)

argparser = ArgumentParser()
argparser.add_argument("--run_id", type=int, required=True)
args = argparser.parse_args()
run_id = args.run_id


def process_chunk(filenames):
    detector = PromptDetector()
    detector.add_heuristic(used_langchain_tool_class)
    detector.add_heuristic(used_langchain_tool)
    detector.add_heuristic(used_in_langchain_llm_call)
    detector.add_heuristic(used_in_openai_call)
    detector.add_heuristic(used_chat_function)
    detector.add_heuristic(used_prompt_or_template_name)
    # detector.add_heuristic(new_line_in_string)
    # detector.add_heuristic(all_strings)

    detector.detect_prompts(filenames, run_id)


def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def run_all(filenames: list[str], n=8):
    os.makedirs(f"{run_id:03d}", exist_ok=True)

    filenames_batched = batched(filenames, len(filenames) // n)
    with Pool(n) as p:
        p.map(process_chunk, filenames_batched)

    data = {}
    for filename in glob(f"{run_id:03d}/prompts-*.json"):
        with open(filename) as f:
            data |= json.load(f)

    with open(f"repo_data_export_{run_id:03d}.json", "w") as w:
        json.dump(data, w, indent=False, ensure_ascii=False)

    shutil.rmtree(f"{run_id:03d}")


if __name__ == "__main__":
    root_dir = "data/scraping-2.0/repos"

    paths = []
    for root, path, files in os.walk(root_dir):
        for file in files:
            paths.append(os.path.join(root, file))

    run_all(paths)
