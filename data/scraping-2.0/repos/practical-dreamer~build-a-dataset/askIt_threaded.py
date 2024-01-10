import os
import openai
import json
import time
import argparse
import re
from atomicwrites import atomic_write
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading
import random

#TODO: incorporate non-threaded features such as include previous chat and verbose messages to terminal

def process_entry_thread(i, entry, args, output_data_len):
    # Lock is used to prevent interleaving of print statements from different threads
    print_lock = threading.Lock()
    thread_id = threading.get_ident()

    with print_lock:
        print(f"askIt_threaded: [Thread-{thread_id}] messages_id {entry['messages_id']} PROCESSING ({i+1}/{output_data_len})")

    messages = list(entry['messages'])

    while entry.get('messages_complete') != True:
        try:
            sleep_time = random.uniform(1, 5)  
            time.sleep(sleep_time)
            
            userPrompt = (messages[-2]['content'])
            assistantPrompt = (messages[-1]['content'])

            response = openai.ChatCompletion.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                top_p=args.top_p,
                presence_penalty=args.presence_penalty,
                frequency_penalty=args.frequency_penalty,
                max_tokens=args.max_tokens
            )
            assistantResponse = response.choices[0].message["content"]

            first_5_words = ' '.join(userPrompt.split()[:5])
            if first_5_words in assistantResponse:
                time.sleep(15)
                continue

            if assistantResponse.startswith(assistantPrompt):
                assistantResponse = assistantResponse[len(assistantPrompt):]

            messages[-1]['content'] += assistantResponse
            entry['messages_complete'] = True            
            with print_lock:
                print(f"askIt_threaded: [Thread-{thread_id}] messages_id {entry['messages_id']} COMPLETE ")
        
        except Exception as e:
            print(f"askIt_threaded: [Thread-{thread_id}] messages_id {entry['messages_id']} ERROR: {e} (Retrying in 5-15 seconds...)")
            error_sleep_time = random.uniform(5, 15)
            time.sleep(error_sleep_time)
    return entry

def main(args):
    openai.api_key = args.api_key
    openai.api_base = args.api_url

    if args.resume:
        if os.path.isfile(args.output_json):
            print(f"askIt_threaded: Existing {args.output_json} file found, attempting to resume...")
            args.input_json = args.output_json

    with open(args.input_json, 'r') as input_file:
        input_data = json.load(input_file)

    output_data = input_data
    incomplete_entries = [i for i, entry in enumerate(output_data) if entry.get('messages_complete') != True]

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_threads) as executor:
        futures = []
        for i in incomplete_entries:
            output_data_len = len(output_data)
            entry = output_data[i]
            futures.append(executor.submit(process_entry_thread, i, entry, args, output_data_len))

        for future in concurrent.futures.as_completed(futures):
            entry = future.result()
            index = next(i for i, x in enumerate(output_data) if x['messages_id'] == entry['messages_id'])
            output_data[index] = entry
            print(f"askIt_threaded: [MAIN] writing to {args.output_json}")
            with atomic_write(args.output_json, overwrite=True) as f:
                json.dump(output_data, f, indent=4)
    print(f"askIt_threaded: Successfully Completed {args.output_json} with " + str(len(output_data)) + " entries.")

if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser(description='OpenAI chat bot')
    parser.add_argument("-input_json", help="Input JSON file", type=str, required=True)
    parser.add_argument("-output_json", help="Output JSON file", type=str)
    parser.add_argument("-include_chat_history", help="Include chat history in subsequent messages", action='store_true')
    parser.add_argument("-max_chat_history", help="Maximum number of elements to keep in chat_history", type=int, default=10)
    parser.add_argument("-resume", help="Resume processing using the output file as the input file", action='store_true')
    parser.add_argument("-max_threads", help="Maximum number of threads", type=int, default=1)
    
    parser.add_argument("-api_key", help="OpenAI API key", type=str, default=os.getenv('OPENAI_API_KEY')) # Get API key from environment variable
    parser.add_argument("-api_url", help="OpenAI API URL", type=str, default=os.getenv('OPENAI_API_URL')) # Get API key from environment variable
    parser.add_argument("-model", help="OpenAI model to use", type=str, default="gpt-3.5-turbo")
    parser.add_argument("-temperature", type=float, default=None)
    parser.add_argument("-top_p", type=float, default=None)
    parser.add_argument("-presence_penalty", type=float, default=0)
    parser.add_argument("-frequency_penalty", type=float, default=0)
    parser.add_argument("-max_tokens", type=int, default=1024)
    args = parser.parse_args()
    #import shlex;args = parser.parse_args(shlex.split("-input_json character_pair_scenarios_prompted -temperature 1.0 -top_p 0.9 -presence_penalty 0.3 -frequency_penalty 0.2 -max_tokens 3000 -model gpt-4 -resume -max_threads 2"))

    # hack to remove new line escapes the shell tries to put in on parameters
    for arg in vars(args):
        value = getattr(args, arg)
        if isinstance(value, str):
            setattr(args, arg, value.replace('\\n', '\n'))

    # Ensure input filename ends with .json
    if args.input_json and not args.input_json.endswith('.json'):
        args.input_json += '.json'

    # If output_json argument is not provided, use the input filename with modified suffix
    if args.input_json and not args.output_json:
        args.output_json = re.sub(r'_([^_]*)$', '_asked', args.input_json)

    # Ensure output filename ends with .json
    if not args.output_json.endswith('.json'):
        args.output_json += '.json'

    # prepend 'jsons/' to the input and output JSON file paths
    args.input_json = os.path.join('jsons', args.input_json)
    args.output_json = os.path.join('jsons', args.output_json)
    
    main(args)
