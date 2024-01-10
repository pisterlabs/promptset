import requests
import os
from dotenv import load_dotenv
import json
from openai import OpenAI
import time

# Load the environment variables from .env file
load_dotenv()

def get_drug_info(indication, limit=10):
    """
    Retrieves drug information for a given indication from the OpenFDA API.
    
    :param indication: The medical condition or symptom to query.
    :param limit: The maximum number of results to return.
    :return: A JSON object with the drug information or an error message.
    """
    api_key = os.getenv('OPENFDA_API_KEY')
    base_url = 'https://api.fda.gov/drug/label.json'
    query = f'indications_and_usage:{indication}'
    request_url = f'{base_url}?search={query}&limit={limit}&api_key={api_key}'
    
    response = requests.get(request_url)
    
    if response.status_code == 200:
        return response.json()
    else:
        return f'Error: {response.status_code} - {response.text}'


def submit_message(assistant_id, thread, user_message):
    """
    Submits a user message to the assistant and triggers a new run.
    
    :param assistant_id: ID of the assistant to use.
    :param thread: The thread object to use.
    :param user_message: The message from the user to the assistant.
    :return: The run object created by triggering the assistant.
    """
    client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_message
    )
    return client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )


def get_response(thread):
    """
    Retrieves all messages from a thread in ascending order.
    
    :param thread: The thread object to retrieve messages from.
    :return: A list of message objects.
    """
    return client.beta.threads.messages.list(thread_id=thread.id, order="asc")


def create_thread_and_run(assistant, client, user_input):
    """
    Creates a new thread and submits the initial user message to the assistant.
    
    :param assistant: The assistant object to use.
    :param client: The OpenAI client object.
    :param user_input: The initial message from the user to the assistant.
    :return: The thread and run objects created.
    """
    thread = client.beta.threads.create()
    run = submit_message(assistant.id, thread, user_input)
    return thread, run


def pretty_print(messages):
    """
    Prints the messages in a thread in a readable format.
    
    :param messages: A list of message objects to print.
    """
    print("# Messages")
    for m in messages:
        print(f"{m.role}: {m.content[0].text.value}")
    print()


def wait_on_run(run, thread):
    """
    Polls the status of a run until it's no longer in progress.
    
    :param run: The run object to monitor.
    :param thread: The thread object associated with the run.
    :return: The updated run object.
    """
    while run.status in ["queued", "in_progress"]:
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run


def make_assistant(indication):
    """
    Creates a new assistant and uploads the necessary data file.
    
    :param indication: The medical condition or symptom to focus on.
    :return: The assistant and client objects created.
    """
    results = get_drug_info(indication)

    with open('data/dataset.json', 'w') as outfile:
        json.dump(results, outfile)        

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    assistant = client.beta.assistants.create(
        name="Clinical Data Assistant medium article",
        instructions="Answer only using the file(s) provided.",
        model="gpt-4-1106-preview",
    )

    file = client.files.create(
        file=open("data/dataset.json", "rb"),
        purpose="assistants",
    )

    assistant = client.beta.assistants.update(
        assistant.id,
        tools=[{"type": "retrieval"}, {"type": "code_interpreter"}],
        file_ids=[file.id],
    )
    
    return assistant, client


def ask_assistant(assistant, client, user_input):
    """
    Asks the assistant a question and prints the response.
    
    :param assistant: The assistant object to use.
    :param client: The OpenAI client object.
    :param user_input: The question to ask the assistant.
    """
    thread, run = create_thread_and_run(assistant, client, user_input)
    run = wait_on_run(run, thread)
    messages = get_response(thread)
    pretty_print(messages)


if __name__ == '__main__':
    assistant, client = make_assistant('prostatitis')
    ask_assistant(assistant, client, 'What are the most effective treatments for prostatitis?')
