import time
import configparser
import os
import openai
import json
import csv

__OUTPUT_DIR = 'output_pdf'
__CONTENT_PROMPT_LENGTH = 4000
MAX_RETRIES = 5  # Maximum number of retries for each API call
RETRY_DELAY = 10  # Delay in seconds before retrying

# Read the CSV file and extract the topic-question pairs
def read_csv_pairs(filename):
    with open(filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        pairs = [(row['topic'], row['question']) for row in reader]
    return pairs

# Read the last processed state
def read_last_processed_state():
    if os.path.exists('last_processed_state.txt'):
        with open('last_processed_state.txt', 'r') as f:
            return f.read().strip()
    return None

# Save the last processed state
def save_last_processed_state(state):
    with open('last_processed_state.txt', 'w') as f:
        f.write(state)

# Updated prompt template for summarization
summary_prompt_template = """
Op basis van de volgende inhoud, vat het antwoord samen op de vraag: "{question}" en citeer relevante delen waarop het antwoord is gebaseerd.

Inhoud:
{combined_content}
"""

config_parser = configparser.ConfigParser()
config_parser.read('openaiconfig.ini')
openai_secret = config_parser['keys']['openaikey']
openai.api_key = openai_secret

topic_question_pairs = read_csv_pairs('query.csv')
last_processed_state = read_last_processed_state()

for combined_file in os.listdir(__OUTPUT_DIR):
    if "-combined.txt" in combined_file:
        parts = combined_file.split('-')
        base_file_name = parts[0]
        topic = '-'.join(parts[1:-1])  # This accounts for topics that might have '-' in them
        combined_filename = os.path.join(__OUTPUT_DIR, combined_file)

        # Find the question associated with the topic
        question = next((q for t, q in topic_question_pairs if t == topic), None)
        if not question:
            print(f"No question found for topic: {topic}. Skipping...")
            continue

        # Check if the combined file exists
        if os.path.exists(combined_filename):
            with open(combined_filename, 'r', encoding='utf-8') as f:
                combined_content = f.read()

            # Create the prompt for summarization
            prompt = summary_prompt_template.format(question=question, combined_content=combined_content)

            summary_file = os.path.join(__OUTPUT_DIR, f"{base_file_name}-{topic}-summary.txt")

            # Check if the summary file already exists
            if not os.path.exists(summary_file):
                for attempt in range(MAX_RETRIES):
                    try:
                        print('calling OpenAI for summarization')
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo-16k",
                            messages=[
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ],
                            temperature=0.5,
                            max_tokens=12000,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0
                        )

                        print('writing summarized response')
                        with open(summary_file, 'w', encoding='utf-8') as f:
                            result = response['choices'][0]['message']['content']
                            f.write(result)
                        break  # If the API call is successful, break out of the retry loop
                    except openai.error.RateLimitError as e:
                        print(f"Rate limit error: {e}. Waiting for a few seconds before retrying...")
                        time.sleep(RETRY_DELAY)
                    except openai.error.Timeout:
                        print("Request timed out. Waiting for a few seconds before retrying...")
                        time.sleep(RETRY_DELAY)
            else:
                print(f"Summary file {summary_file} exists, skipping")
        else:
            print(f"Combined file {combined_filename} does not exist, skipping")

print("Summarization completed!")
