import os
import json
import threading
import shutil
import argparse
import openai
from tqdm import tqdm
import time
import pickle
from nltk import sent_tokenize

openai.api_key = 'your_OpenAI_api_key_here'


def remove_last_sentence(paragraph):
    """
    Input: A paragraph of text
    Output: The paragraph with the first and last sentence removed

    """
    sentences = sent_tokenize(paragraph)
    if len(sentences) < 3:
        return paragraph
    return " ".join(sentences[0:len(sentences) - 1])


def gpt_new_conversation(prompt, temperature=0, pp=0):
    """
    Input: The prompt temperature (default 0) and presence penalty (default 0)
    Output: A list of dict that contains the user prompt and ChatGPT's answer to the prompt
    """

    try:
        conversation = [{'role': 'user', 'content': prompt}]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=conversation,
            temperature=temperature,
            request_timeout=30,
            presence_penalty=pp,
            max_tokens=310,
            top_p=1,
            frequency_penalty=0)
        content = response.choices[0]['message']['content']
        # Remove the first sentence of the answer
        role = response.choices[0]['message']['role']
        conversation.append({'role': role, 'content': content})
        # This can be useful to calculate the cost
        # cost = response.usage['total_tokens']

        return conversation
    except:
        print("*** ChatGPT has time out. If you see this message too many times, please consider stop ***")
        time.sleep(5)
        return gpt_new_conversation(prompt, temperature)


def gpt_annotate_req_abs_pair(req, abs, first_prompt, last_prompt, removal):
    """
    Input: req: The aspect in the question pair
           abs: The abstract in the question pair
           first_prompt: The prompt template for ChatGPT
           last_prompt: The latter prompt template for ChatGPT (if needed)
           removal: A boolean that determines whether the first/last sentence in the reasoning should be removed

    Output: A list of dictionary that records conversation with ChatGPT
    """
    prompt_1 = first_prompt.format(req=req, abstract=abs)
    result_conversation = gpt_new_conversation(prompt_1)

    return result_conversation


def annotate(req_abs_need_to_input, output_file_name, curr_thread_id, curr_config):
    """
    Input: req_abs_need_to_input: A list of dictionary that contains the question pairs to annotate
           output_file_name: The output file to store the annotation results
           curr_thread_id: The thread id that calls this function
           curr_config: A key for the config dictionary that decides the prompt template

    Output: No output. Instead, this function stores the annotation to the output_file_name
    """
    first_prompt = curr_config['initial']
    last_prompt = curr_config['later']
    removal = curr_config['removal']

    result = []
#     for d in tqdm(req_abs_need_to_input,desc = "In thread", leave = False):
    for d in req_abs_need_to_input:
        curr_req = d['aspect']
        curr_abs = d['abstract']
        result.append(gpt_annotate_req_abs_pair(curr_req, curr_abs, first_prompt, last_prompt, removal))
        # time.sleep(3)

    with open(output_file_name, 'w') as f:
        json.dump(result, f)

# This Python script utilizes the OpenAI GPT-3 model to generate annotations for given question pairs in a json file.
# The main program uses multithreading for efficient processing and takes in various arguments such as the input query file,
# output file, configuration of the prompt, the number of threads to use, and the range of indices to consider from the input file.
#
# Input: A JSON file with research queries.
# Output: A JSON file with annotations generated for the research queries.
#
# The script expects the following command-line arguments:
# -q/--query_file: JSON file with the queries to pass.
# -o/--output_file: File to write the responses to.
# -p/--config_path: Path to the configuration file.
# -c/--config: Configuration of the prompt.
# -t/--thread_num: Number of threads to use for the execution.
# -s/--index_to_start: Index in the query file to start processing from.
# -e/--index_to_end: Index in the query file to stop processing at.
#
# The 'config_path' should point to a file containing a dictionary of configurations where each configuration consists of
# 'initial', 'later' and 'removal' keys.
#
# Note: Please ensure to set your OpenAI API key before running the script.


if __name__ == "__main__":
    """
    This program will annotate the question pairs using ChatGPT 3.5
    """

    parser = argparse.ArgumentParser(
        prog='ChatGPT',
        description='Takes in json file of research queries and outputs relevant ideas',
        epilog='Created by LEI'
    )

    parser.add_argument('-q', '--query_file', required=True, help='Queries to pass in json format')
    parser.add_argument('-o', '--output_file', required=True, help='Output file to write responses to')
    parser.add_argument('-p', '--config_path', required=True, help='path to the config file')
    parser.add_argument('-c', '--config', required=True, help='Configuration of the prompt')
    parser.add_argument('-t', '--thread_num', required=True, help='Number of threads used in the program')
    parser.add_argument('-s', '--index_to_start', required=True, help='The index to start in the query file')
    parser.add_argument('-e', '--index_to_end', required=True, help='The index to end in the query file')

    input_path = ""
    output_path = "./annotation_results/"

    args = parser.parse_args()
    prompt_path = args.config_path
    start = int(args.index_to_start)
    end = int(args.index_to_end)
    thread_num = int(args.thread_num)
    query_file = args.query_file
    query_file = input_path + query_file
    output_name = args.output_file + f"_{start}_{end}" + "/"
    output_path = output_path + output_name

    result_file_name = output_path + 'combined_result.json'

    start_time = time.time()
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    with open(query_file, 'rb') as f:
        input_list = pickle.load(f)

    input_list = input_list[start:end]

    index_file_name = output_path + 'index_of_results.pickle'

    index_list = []
    for i in input_list:
        curr_index_tuple = (i['aspect_id'], i['abstract_id'])
        index_list.append(curr_index_tuple)

    with open(index_file_name, 'wb') as f:
        pickle.dump(index_list, f)

    anno_per_thread = len(input_list) / thread_num
    print(f"{len(input_list)} annotations needed")
    print(f"Using {thread_num} threads")
    print(f"{anno_per_thread} annotations for each thread")
    print(f"The input file is {query_file}")
    print(f"The output files' will start with {output_path}")

    with open(prompt_path, 'rb') as f:
        config_dict = pickle.load(f)

    curr_config = config_dict[args.config]

    thread_list = []
    output_name_list = []
    thread_id = 0
    for i in range(thread_num):
        start = int(anno_per_thread * i)
        end = int(anno_per_thread * (i + 1))

        #             print(f"Thread {i}: start at {start}, end at {end}")
        output_name = output_path + "result_" + str(start) + "_" + str(end) + ".json"
        output_name_list.append(output_name)
        thread_list.append(
            threading.Thread(target=annotate, args=(input_list[start:end], output_name, thread_id, curr_config)))
        thread_id += 1
    for t in thread_list:
        t.start()
    #             time.sleep(0.5)

    for i in range(thread_num):
        thread_list[i].join()
        # print(f"Thread {i} finished.")

    print("All thread ends")

    final_output = []
    for name in output_name_list:
        with open(name, 'r') as f:
            curr_data = json.load(f)

        final_output += curr_data

    with open(result_file_name, 'w') as f:
        json.dump(final_output, f)

    print("Time used:")
    print(time.time() - start_time)
