import sys
import json
import os
import shutil
from argparse import ArgumentParser
from glob import glob
from itertools import islice
from functools import partial
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

    return detector.detect_prompts(filenames, run_id)

if __name__ == '__main__':
    assert len(sys.argv) == 2
    file_paths = sys.argv[1]
    file_paths = file_paths.split('~~~')
    print('file:', file_paths)
    # print(process_chunk([file_paths], 0))
    sys.stdout.flush()
