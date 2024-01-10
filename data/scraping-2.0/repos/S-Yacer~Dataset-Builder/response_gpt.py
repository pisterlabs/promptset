import os
import openai
import json
import time
import argparse
import re
from atomicwrites import atomic_write
import concurrent.futures
import threading
import random

'''Main script to generate responses from GPT, contains some pre-processing for quality of life'''

# Global variable to ensure unique message ids across the threads
next_message_id = 1

# Pre-Processing functions
def format_filename(filename, default_suffix):
    if not filename.endswith('.json'):
        filename += '.json'
    # Ensure we don't duplicate the 'jsons' directory in the path
    if not filename.startswith('jsons'):
        filename = os.path.join('jsons', filename)
    return filename

def clean_empty_assistant_entries(data):
    cleaned_data = []
    for entry in data:
        entry['messages'] = [msg for msg in entry['messages'] if not (msg['role'] == 'assistant' and msg['content'].strip() == '')]
        # Check if there are any assistant messages left
        assistant_messages = [msg for msg in entry['messages'] if msg['role'] == 'assistant']
        # If there are assistant messages, append the entry to the cleaned data
        if assistant_messages: 
            cleaned_data.append(entry)
    return cleaned_data

# Sort data entries based on the messages_id
def sort_by_message_id(data):
    return sorted(data, key=lambda x: int(x['messages_id']))

# Main function
def process_single_entry(i, rep, entry, args):
    global next_message_id
    print_lock = threading.Lock()
    thread_id = threading.get_ident()

    with print_lock:
        print(f"[Thread-{thread_id}] messages_id {entry['messages_id']}_{rep} PROCESSING")

    new_entry = entry.copy()
    new_entry['messages_id'] = f"{str(next_message_id).zfill(5)}"
    next_message_id += 1

    messages = list(new_entry['messages'])
    # The main loop ensures retries until a message is considered complete
    while not new_entry.get('messages_complete'):
        sleep_time = random.uniform(1, 5)
        time.sleep(sleep_time)
        try:
            # Generate response from GPT
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
            # Append the assistant's response to the messages list if it's not empty
            if assistantResponse.strip() != '':  
                messages.append({
                    "role": "assistant",
                    "content": assistantResponse
                })
            new_entry['messages'] = messages
            new_entry['messages_complete'] = True

        except Exception as e:
            with print_lock:
                print(f"[Thread-{thread_id}] messages_id {entry['messages_id']}_{rep} ERROR: {e} (Retrying in 5-15 seconds...)")

    with print_lock:
        print(f"[Thread-{thread_id}] messages_id {entry['messages_id']}_{rep} COMPLETE ")

    return new_entry


def main(args):
    openai.api_key = "OPENAI_API_KEY"
    openai.api_base = "https://api.openai.com/v1"

    with open(args.input_json, 'r') as input_file:
        input_data = json.load(input_file)

    output_data = input_data
    incomplete_entries = [i for i, entry in enumerate(output_data) if not entry.get('messages_complete')]

    # Use threading to process multiple entries concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_threads) as executor:
        futures = [executor.submit(process_single_entry, i, rep, output_data[i], args) for i in incomplete_entries for rep in range(args.num_responses)]
        for future in concurrent.futures.as_completed(futures):
            new_entry = future.result()
            # Update the output data
            output_data.append(new_entry)
            output_data = clean_empty_assistant_entries(output_data)
            output_data = sort_by_message_id(output_data)
            with atomic_write(args.output_json, overwrite=True) as f:
                json.dump(output_data, f, indent=4)
    print(f"Successfully Completed {args.output_json} with {len(output_data)} entries.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenAI chat bot')
    # Define cli arguments
    parser.add_argument("-input_json", help="Input JSON file", type=str, required=True)
    parser.add_argument("-output_json", help="Output JSON file", type=str, default=None)
    parser.add_argument("-max_threads", help="Maximum number of threads", type=int, default=1)
    parser.add_argument("-num_responses", help="Number of responses per prompt", type=int, default=1)
    parser.add_argument("-model", help="OpenAI model to use", type=str, default="gpt-3.5-turbo")
    parser.add_argument("-temperature", type=float, default=None)
    parser.add_argument("-top_p", type=float, default=None)
    parser.add_argument("-presence_penalty", type=float, default=0)
    parser.add_argument("-frequency_penalty", type=float, default=0)
    parser.add_argument("-max_tokens", type=int, default=1024)

    args = parser.parse_args()
    # Adjust filenames and handle suffixes
    args.input_json = format_filename(args.input_json, '_asked')
    args.output_json = format_filename(args.output_json if args.output_json else re.sub(r'_([^_]*)$', '_asked', args.input_json), '')
    
    main(args)