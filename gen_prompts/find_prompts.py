import json
import os
import shutil
from argparse import ArgumentParser
from glob import glob
from itertools import islice
from functools import partial
from multiprocessing import Pool
from gen_prompts.parsers import (
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


def process_chunk(filenames, run_id: int):
    detector = PromptDetector()
    detector.add_heuristic(used_langchain_tool_class)
    detector.add_heuristic(used_langchain_tool)
    detector.add_heuristic(used_in_langchain_llm_call)
    detector.add_heuristic(used_in_openai_call)
    detector.add_heuristic(used_chat_function)
    detector.add_heuristic(used_prompt_or_template_name)
    detector.add_heuristic(find_from_file)
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


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--run_id", type=int, required=True)
    argparser.add_argument("--repo_dir", type=str, default="data/scraping-2.0/repos")
    argparser.add_argument("--threads", type=int, default=8)
    args = argparser.parse_args()
    os.makedirs(f"{args.run_id:03d}", exist_ok=True)

    # Find all files
    paths = []
    for root, path, files in os.walk(args.repo_dir):
        for file in files:
            paths.append(os.path.join(root, file))

    # Batch into thread-count batches, and apply the heuristics
    filenames_batched = batched(paths, len(paths) // args.threads)
    with Pool(args.threads) as p:
        p.map(partial(process_chunk, run_id=args.run_id), filenames_batched)

    # Join the thread data by loading output files
    data = {}
    for filename in glob(f"{args.run_id:03d}/prompts-*.json"):
        with open(filename) as f:
            data |= json.load(f)

    # save joined data
    with open(f"data/repo_data_export_{args.run_id:03d}.json", "w") as w:
        json.dump(data, w, indent=False, ensure_ascii=False)

    # Clean folder
    shutil.rmtree(f"{args.run_id:03d}")
