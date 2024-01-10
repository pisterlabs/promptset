"""
This script reads movie review files and returns a CSV.
The CSV has the filename, the polarity of the review, and the cost for gpt to label the review.
"""

import argparse
import time
import csv
import json
import os
import sys
import glob
import openai

MODEL = "gpt-3.5-turbo-0301"
CSV_FILENAME = "sentiment_labels.csv"

def main():
    setup_openai()
    args = process_args()
    files_to_process = get_review_files() - get_already_processed_files()
    progress_ticker = generate_progress_ticker(len(files_to_process)) if args.progress else None

    # Prepare the output file.
    if not file_exists(CSV_FILENAME):
        write_header()

    with open(CSV_FILENAME, "a") as csv_file:
        for filename in files_to_process:
            if progress_ticker is not None:
                next(progress_ticker)
            try:
                process_file(filename, csv_file)
            except KeyboardInterrupt:
                print("Keyboard interrupt detected. Goodbye")
                sys.exit(1)
            except Exception as e:
                print(f"An exception of type {type(e).__name__} occurred while processing '{filename}': {str(e)}")
                continue
        if progress_ticker is not None:
            next(progress_ticker)

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress", help="Shows processing progress information during runtime", action="store_true")
    return parser.parse_args()

def setup_openai():
    openai.api_key = os.environ.get("OPENAI_API_KEY")

def get_already_processed_files():
    already_processed = set()
    if file_exists(CSV_FILENAME):
        with open(CSV_FILENAME, "r") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                already_processed.add(row["filename"])
    return already_processed

def file_exists(filename):
    """Predicate that indicates if the file exists."""
    return os.path.exists(filename)

def get_review_files():
    return set(glob.glob("*.txt"))

def generate_progress_ticker(total_count):
    last_message_length = 0
    for i in range(total_count):
        message = f"Processing {i+1} of {total_count}"
        print(message, end="\r")
        last_message_length = len(message)
        yield
    print("\r" + " " * last_message_length, end="\r")
    print("Done.")
    yield

def write_header():
    header = "filename,sentiment,cost\n"
    with open(CSV_FILENAME, "w") as output:
        output.write(header)

def process_file(filename, output_file):
        with open(filename, "r") as f:
            review_data = f.read()

        def gpt_call():
            return openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Respond with 'postive' or 'negative'"},
                    {"role": "user", "name": "sentiment-tester", "content": review_data},
                ],
            )
        gpt_response = handle_gpt_exceptions(gpt_call)

        basename, _ = os.path.splitext(filename)
        with open(f"{basename}-gpt.json", "w") as response_file:
            response_file.write(json.dumps(gpt_response))

        sentiment = gpt_response["choices"][0]["message"]["content"]
        cost = gpt_response["usage"]["total_tokens"]
        output_file.write(f"{filename},{sentiment},{cost}\n")

def handle_gpt_exceptions(call):
    """
        Makes the chat gpt call, and handles exceptions if it fails.
        For certain exceptions, it will apply exponential backoff and try again.
    """
    num_retries = 10
    wait_time = 3
    for _ in range(num_retries):
            # GPT limits 20 calls per minute. This sleep ensures that limit is not hit.
        time.sleep(3)
        try:
            response = call()
        except openai.error.RateLimitError:
            print(f"Rate limit error hit. Sleeping for {wait_time} seconds, then trying again.")
            time.sleep(wait_time)
            wait_time *= 2
            continue
        return response
    raise OutOfRetries()

class OutOfRetries(Exception):
    def __init__(self):
        self.message = "Ran out of retries"
    def __str__(self):
        return self.message

if __name__ == "__main__":
    main()
