#Test GSM8K socratic full (train) or test data against GPT3.5 and GPT4
import json
import openai
import csv
import os
import time
import concurrent.futures
import signal

# Configure the OpenAI API key
openai.api_key = 'key-go-here-bro'

# File paths
input_file_path = 'test_socratic.jsonl'
output_csv_path = 'gpt_responses.csv'

# Function to send a question to GPT and get the response
def get_gpt_response(question, model_name, retries=3):
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[{"role": "system", "content": "You are a taking a math test."},
                          {"role": "user", "content": question}],
                timeout=10
            )
            return response.choices[0].message['content'].strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"Final attempt failed for question: {question}\nError: {e}")
                return None

# Function to parse the Socratic method and correct answer
def parse_answer(answer):
    step_by_step, correct_answer = answer.split("####")
    return step_by_step.strip(), correct_answer.strip()

# Function to handle a single question
def process_question(entry, model_name):
    question = entry['question']
    socratic_method, correct_answer = parse_answer(entry['answer'])
    model_response = get_gpt_response(question, model_name)

    return [question, socratic_method, correct_answer, model_name, model_response]

# Function to write a single record to CSV
def write_to_csv(record, problem_id):
    with open(output_csv_path, 'a', newline='', encoding='utf-8') as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerow([problem_id] + record)

# Function to handle user interruption
def signal_handler(sig, frame):
    print("\nProcess interrupted by the user. Exiting...")
    exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Main function to process the dataset
def process_dataset(model_name):
    file_exists = os.path.isfile(output_csv_path)
    if not file_exists:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as outfile:
            csv_writer = csv.writer(outfile)
            csv_writer.writerow(['ProblemID', 'Question', 'Socratic Method', 'Correct Answer', 'Model', 'Model Response'])

    with open(input_file_path, 'r') as infile:
        entries = [json.loads(line) for line in infile]

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_question = {executor.submit(process_question, entry, model_name): i+1 for i, entry in enumerate(entries)}

        for future in concurrent.futures.as_completed(future_to_question):
            problem_id = future_to_question[future]
            try:
                record = future.result()
                write_to_csv(record, problem_id)
                print(f"Processed Problem ID: {problem_id}")
            except Exception as e:
                print(f"Error processing Problem ID: {problem_id}: {e}")

# Run the main function with GPT-3.5
print("Processing with GPT-3.5...")
process_dataset("gpt-3.5-turbo")

# Ask user if they want to proceed with GPT-4
proceed = input("Do you want to proceed with GPT-4? (y/n): ").lower()
if proceed == 'y':
    print("Processing with GPT-4...")
    process_dataset('gpt-4')
