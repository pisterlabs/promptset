'''
Some basic text intelligence functions

1. Summarization
2. Key Insights
'''

import openai
import tiktoken
import os
from tqdm.auto import tqdm

from time import sleep

import threading

chosen_model = "gpt-4"
print("Using model: ", chosen_model)

def get_num_tokens(input, model="text-davinci-003"):

    encoding = tiktoken.encoding_for_model(model)
    comp_length = len(encoding.encode(input)) #lower bound threshold since it seems that tiktoken is not that accurate

    return comp_length

def summarise(input):

    #init length of the prompt is 216 tokens, rounding up to 300. assume each summary will summarise to half its length at max. so (4096-300)/3*2 = 2530. so the max length of input we give must be 2530
    #bump that down to 2000 ish and we modulo based on that. 1500 works better for now and works within our context length.
    total_summary = ""

    input = input.replace("\n", ".")

    sentences = input.split(".")

    num_tokens = 0
    curr_corpus = ""

    threads = []
    result_container = {}

    for sentence in sentences:

        sentence += "."  # reappend the period
        curr_corpus += sentence
        num_tokens += get_num_tokens(sentence)

        if num_tokens > 3500:
            # Create a thread object to execute summarise_helper() in a separate thread
            thread_index = len(threads)
            threads.append(threading.Thread(target=summarise_helper, args=(curr_corpus, result_container, thread_index)))

            # Start the worker thread
            threads[-1].start()

            curr_corpus = ""
            num_tokens = 0

    # Account for the last iteration
    if num_tokens > 0:
        thread_index = len(threads)
        threads.append(threading.Thread(target=summarise_helper, args=(curr_corpus, result_container, thread_index)))

        # Start the worker thread
        threads[-1].start()
    
    print("Waiting for summarization threads to finish... Number of threads are: ", len(threads))

    # Join all threads
    for i, worker_thread in tqdm(enumerate(threads)):
        worker_thread.join()
        total_summary += result_container[i]

    return total_summary
    
def summarise_helper(input, result_container, index):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    #TODO: increase the length of the output through prompt engineering
    #The other more naive solution is to reduce the input size to 1000 tokens. although depending on how good the prompt engineering is this summarizer might be able to account for overlapping data in different texts
    start_prompt = "You are SummarizerGPT. You create summaries that keep all the information from the original text. You must keep all numbers and statistics from the original text. You will provide the summary in succint bullet points. For longer inputs, summarise the text into more bullet points. You will be given a information, and you will give me a bulleted point summary of that information."
    
    ask_prompt = """Summarise the following text for me into a list of bulleted points.
    
    Information:
    
    {information}""".format(information=input)

    ask_prompt = start_prompt + "\n" + ask_prompt

    #introducing exponential backoff with max 10 retries
    try_count = 1
    while try_count <= 10:
        try:
            response = openai.ChatCompletion.create(
                model=chosen_model,
                messages=[
                    {"role": "system", "content": start_prompt},
                    {"role": "user", "content": ask_prompt}
                ]
            )
            print(f"Summarisation Thread {index} finished successfully")
            break
        except Exception as e:
            rest_time = try_count * 10
            print(f"Error in Summarisation Thread {index}: {e}")
            print(f"Retrying in {rest_time}...")
            sleep(rest_time)
            try_count += 1
            continue


    result_container[index] = response.choices[0].message.content
    # return response.choices[0].message.content