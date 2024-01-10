import csv
from openai import OpenAI
import os
import time

# initialising variables
TEMPERATURE = 0.0               # 0.0 0.25 0.5 0.75 1.0

# ONLY CHANGE THE ABOVE VARIABLE FOR A CERTAIN TEMPERATURE FOR CONTINUATION OF ChatGPT gpt-3.5-turbo


MODEL_NAME = "gpt-3.5-turbo"
MODEL_DIRECTORY = os.path.join('output_files', MODEL_NAME)
MODEL_NON_JAILBREAK_DIR = os.path.join(MODEL_DIRECTORY, 'non-jailbreak')

REGULAR_QUESTION_SET_PATH = "input_files/control_dataset_continued_" + str(TEMPERATURE) + ".csv"

# function to create a directory if not exists
def create_directory(directory_path):
    # Check if the directory already exists
    if not os.path.exists(directory_path):
        # If not, create the directory
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")

# Create folders if not exists
create_directory(MODEL_DIRECTORY)
create_directory(MODEL_NON_JAILBREAK_DIR)


output_file_name = os.path.join(MODEL_NON_JAILBREAK_DIR, MODEL_NAME + " " + str(TEMPERATURE) + '.csv')

# call evertime a new row is read from the question set to reset a chat
client = OpenAI(
    api_key="API KEY HERE"     
)

regular_question_set_prompts = []

# start reading for the inputs
with open(REGULAR_QUESTION_SET_PATH, encoding='utf-8') as regular_question_set_file:
    regular_question_set_reader = csv.reader(regular_question_set_file)

    # iterating through all possible questions (includes repeat 3 times)
    # each row: input_id, content_policy_id, content_policy_name, q_id,question, response_idx
    for row in regular_question_set_reader:

        # send question as input
        input_prompt = row[4]
    
        # Save the prompts in local memory so I can close the csv file to prevent time out
        regular_question_set_prompts.append([row[0], input_prompt])


# iterate through prompts and send them in as input
for input_prompt in regular_question_set_prompts:

    print('Producing Response')
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0613",
        messages=[{"role": "user", "content": input_prompt[1]}],
        temperature=TEMPERATURE
    )

    output_response = response.choices[0].message.content

    print('Response Received')

        # write response into the output file
    with open(output_file_name, 'a', newline='', encoding='utf-8') as phase_one_output_file:
        phase_one_output_writer = csv.writer(phase_one_output_file)
        phase_one_output_writer.writerow([input_prompt[0], MODEL_NAME, TEMPERATURE, output_response])
        print(input_prompt[0], ': Saved Output ' + output_file_name)

    time.sleep(10)


print("Process Completed")




