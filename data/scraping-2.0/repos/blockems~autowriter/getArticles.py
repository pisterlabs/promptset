import os
import re
import json
import openai
import time

from datetime import datetime

max_retries = 3
retry_delay = 10  # seconds

def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

with open("openaiapikey.txt", "r") as api_key_file:
    openai.api_key = api_key_file.read().strip()

# re-do rate failure
# with open('PCVDTopics.json', 'r') as json_file:
#    topics1 = json.load(json_file)
topics1=[]

with open('HyperAutomationTopics.JSON', 'r') as json_file:
    topics2 = json.load(json_file)

with open("KeywordPrompt.txt", "r") as keyword_prompt_file:
    keyword_prompt_template = keyword_prompt_file.read()

os.makedirs("subjects", exist_ok=True)  # Changed from "articles"
os.makedirs("log", exist_ok=True)

topics = {
    'PCVDTopics': topics1,
    'HyperAutomationTopics': topics2
}

# Counter to limit the run to 2 iterations for testing purposes
counter = 0

for json_name, topic_list in topics.items():
    for topic in topic_list:
        #if counter >= 2:
        #    break

        keyword_prompt = keyword_prompt_template.replace("<<TOPIC>>", topic)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": keyword_prompt}
        ]

        retries = 0
        while retries < max_retries:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.7,
                    top_p=1,
                    max_tokens=1000,
                )
                break  # If successful, exit the retry loop
            except openai.error.RateLimitError:
                if retries < max_retries - 1:
                    print(f"Rate limit error. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retries += 1
                else:
                    raise  # Raise the error if retries exhausted

        assistant_response = response.choices[0].message["content"].strip()
        sanitized_topic = sanitize_filename(topic)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_filename = f"subjects/{timestamp}-{json_name}-{sanitized_topic}.json"  # Changed from "articles"

        with open(output_filename, "w") as output_file:
            json.dump(assistant_response, output_file)  # Save as JSON

        print(f"Saved article to {output_filename}")

        # Save prompt and settings to log directory
        log_filename = f"log/{timestamp}_{json_name}_{sanitized_topic}.txt"

        with open(log_filename, "w") as log_file:
            log_file.write("Prompt:\n")
            log_file.write(keyword_prompt)
            log_file.write("\n\nSettings:\n")
            log_file.write(f"Model: gpt-3.5-turbo\n")
            log_file.write(f"Temperature: 0.7\n")
            log_file.write(f"Top_p: 1\n")
            log_file.write(f"Max tokens: 1000\n")

        # Increment the counter
        counter += 1
