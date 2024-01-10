import openai
import os
from dotenv import load_dotenv
import pandas as pd
import time

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Create an instance of the OpenAI client
client = openai.OpenAI(api_key=api_key)


def get_industries(df):
    # Iterate through each problem statement and generate industry prediction
    # df = df.head(20)
    industries = []
    problem_column = 'problem'
    industry_column = 'industry'
    i = 0
    file_path = 'assistant_instructions.txt'
    file_content = ""

    # Open the file in read mode
    with open(file_path, 'r', encoding='utf-8') as file:
        # Read the entire content of the file into a string variable
        file_content = file.read()
    assistant = client.beta.assistants.create(
        name="Industry Prediction Assistant",
        instructions=file_content,
        tools=[{"type": "retrieval"}],
        model="gpt-4-1106-preview"
    )
    assistant_id = assistant.id
    thread = client.beta.threads.create()
    thread_id = thread.id
    print("Thread ID: " + thread_id)
    problem_statements = ""
    i = 0
    for problem in df[problem_column]:
        # if i ==2:
        #     break
        # else:
        #     i += 1
        try:
            print(problem)
            problem_statements += problem + "\n"
        except TypeError:
            # This means that we have reached EOF and break out of for loop
            break

    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content="Find the industries in these problem statements: \n" + problem_statements,
    )
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        instructions="Only give the industry names and nothing else - not even the word 'industry' - try to limit it to two words at max. Return a list of the industries separated by commas. If a problem statement has two industries, list both and separate them with an &."
    )

    # If run is 'completed', get messages and print
    while True:
        # Retrieve the run status
        run_status = client.beta.threads.runs.retrieve(
            thread_id=thread.id, run_id=run.id)
        print(run_status.model_dump_json(indent=4))
        time.sleep(10)
        if run_status.status == 'completed':
            messages = client.beta.threads.messages.list(thread_id=thread.id)
            # print("message:")
            # print(messages)
            break
        else:
            # sleep again
            time.sleep(2)
    return messages.data[0].content[0].text.value


def get_metric(df, metric):
    # file_path = 'assistant_instructions.txt'
    # file_content = ""

    # # Open the file in read mode
    # with open(file_path, 'r') as file:
    #     # Read the entire content of the file into a string variable
    #     file_content = file.read()
    # assistant = client.beta.assistants.create(
    #     name="Industry Prediction Assistant",
    #     instructions=file_content,
    #     tools=[{"type": "retrieval"}],
    #     model="gpt-3.5-turbo-1106"
    # )
    assistant_id = "asst_n2ZJaW7Eh3IfAYKtinp9uUs3"
    thread = client.beta.threads.create()
    thread_id = thread.id
    problem_solution_pair = ""
    for index, row in df.iterrows():
        problem_solution_pair += row['problem'] + \
            "\n" + row['solution'] + "\n\n"
    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content="Find the " + metric +
        " of these problem statements and give it a score out of 10: \n" + problem_solution_pair,
    )

    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        instructions=f"Give me the {metric} for the problem-solution pairs below (no need for any files) as a comma-separated string. Don't give me anything else, just the score out of 10" + problem_solution_pair
    )

    # If run is 'completed', get messages and print
    while True:
        # Retrieve the run status
        run_status = client.beta.threads.runs.retrieve(
            thread_id=thread.id, run_id=run.id)
        print(run_status.model_dump_json(indent=4))
        time.sleep(10)
        if run_status.status == 'completed':
            messages = client.beta.threads.messages.list(thread_id=thread.id)
            # print("message:")
            # print(messages)
            break
        else:
            # sleep again
            time.sleep(2)
    return messages.data[0].content[0].text.value


def wait_on_run(run, thread):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run
