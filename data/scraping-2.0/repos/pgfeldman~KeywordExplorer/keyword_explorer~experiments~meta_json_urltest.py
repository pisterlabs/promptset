import re
import requests
import json
import os
import tkinter as tk
import tkinter.filedialog
from keyword_explorer.OpenAI.OpenAIEmbeddings import OpenAIComms

from typing import Dict, List


def count_valid_urls(text):
    # Find all URLs using regular expressions
    urls = re.findall(r'(https?://\S+(?<![:?#=]))+', text)

    # Test each URL and count valid and invalid ones
    valid_count = 0
    invalid_count = 0
    for url in urls:
        try:
            print("testing url {}".format(url))
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                valid_count += 1
            else:
                invalid_count += 1
        except (requests.exceptions.InvalidURL, requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
            invalid_count += 1

    return valid_count, invalid_count

def read_json_files(directory_path:str) -> Dict:
    """
    Reads all the json files in the provided directory and places the
    values of the array labeled "experiments" into a Pandas dataframe.
    Returns a dictionary of dataframes, where the key is the filename.
    """
    all_dicts = {}
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            with open(os.path.join(directory_path, filename), 'r') as file:
                json_data = json.load(file)
                all_dicts[filename] = json_data
    return all_dicts

def write_json_files(directory_path, all_dicts:Dict):
    for key, d in all_dicts.items():
        filename = "url_{}".format(key)
        fullpath = "{}/{}".format(directory_path, filename)
        print("saving {}".format(fullpath))
        with open(fullpath, mode="w", encoding="utf-8") as f:
            json.dump(d, f, indent=4)

def find_patterns(input_string) -> List:
    # pattern = r"\(source \d+\)\."
    pattern = r'\(source (\d{5})\)'
    matches = re.findall(pattern, input_string)
    return [int(match) for match in matches]


def evaluate_data(all_dicts:Dict):
    print("evaluate_data")
    oac = OpenAIComms()
    # go through all the dicts and:
    for name, d in all_dicts.items():
        print("key = {}".format(name))
        # 1) Get the index list from the context and explicitly compare those against the numbers in the "context_response" strings
        context = d['context']
        source_list = find_patterns(context)
        print("\t{}".format(source_list))
        for ed in d['experiments']:
            context_response = ed['context_response']
            match_list = []
            for i in source_list:
                if str(i) in context_response:
                    match_list.append(i)
            ed['index_list'] = match_list
            print("\ted['index_list'] = {}".format(ed['index_list']))
            good_urls, bad_urls = count_valid_urls(context_response)
            ed['ctx_good_urls'] = good_urls
            ed['ctx_bad_urls'] = bad_urls
            print("\ted['ctx_good_urls'] = {}, ed['ctx_good_urls'] = {}".format(ed['ctx_good_urls'], ed['ctx_good_urls']))
            no_context_response = ed['no_context_response']
            good_urls, bad_urls = count_valid_urls(no_context_response)
            ed['no_ctx_good_urls'] = good_urls
            ed['no_ctx_bad_urls'] = bad_urls
            print("\ted['no_ctx_good_urls'] = {}, ed['no_ctx_bad_urls'] = {}".format(ed['no_ctx_good_urls'], ed['no_ctx_bad_urls']))


            prompt = "In the following context, find all the sources and list them:\n\nCONTEXT:\n\n{}\n\nSource list:\n".format(no_context_response)
            no_ctx_r = oac.get_prompt_result_params(prompt, max_tokens=512, temperature=0.75, top_p=1, frequency_penalty=0, presence_penalty=0, engine="gpt-3.5-turbo-0301")
            ed['no_ctx_sources'] = no_ctx_r
            ed['no_ctx_good_sources'] = 0
            ed['no_ctx_bad_sources'] = 0


def main():
    print("meta_json_urltest")
    root = tk.Tk()
    root.withdraw()

    # open the file dialog and get the selected directory
    path = tkinter.filedialog.askdirectory()
    all_dicts = read_json_files(path)

    evaluate_data(all_dicts)

    write_json_files(path, all_dicts)


if __name__ == "__main__":
    main()