import dotenv
import os
import time
from openai import OpenAI
import warnings


# Suppress the specific warning related to relevance scores
warnings.filterwarnings("ignore", category=UserWarning, message="No relevant docs were retrieved using the relevance score threshold 0.65")

# load api key
dotenv.load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()
# defaults to getting the key using os.environ.get("OPENAI_API_KEY")
# if you saved the key under a different environment variable name, you can do something like:
# client = OpenAI(
#   api_key=os.environ.get("CUSTOM_ENV_NAME"),
# )

# List all files in the "inputASD" directory
directory = "inputASD"
file_ids = []
for filename in os.listdir(directory):
    if filename.endswith(".pdf" or ".txt"):
        file_path = os.path.join(directory, filename)
        with open(file_path, "rb") as file:
            uploaded_file = client.files.create(
                file=file,
                purpose='assistants'
            )
            file_ids.append(uploaded_file.id)


# # Add the file to the assistant
# assistant = client.beta.assistants.create(
#   instructions='''
#   Your task is to assist users in understanding different management strategies for Autism. 
#   You should rely exclusively on the specific research files and documents provided to you. 
#   Do not use any information from your training or external sources. 
#   Answer questions based solely on the contents of these provided files. 
#   If a question arises that cannot be answered with the information in these files, respond with 'I don't know.' 
#   Your goal is to ensure that all information given is accurate, research-based, and directly sourced from these documents. 
#   Avoid speculation or guessing. Focus on delivering factual, proven methods as indicated by the uploaded information. 
#   Please confirm your understanding of these instructions.''',
#   model="gpt-4-1106-preview",
#   tools=[{"type": "retrieval"}],
#   file_ids=file_ids  # Use the list of uploaded file IDs
# )

# Your specific assistant ID
ASSISTANT_ID = 'asst_Hg7Keq9V1Fpm0NXdav02mepz'

def wait_on_run(run, thread):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run


def submit_message(assistant_id, thread, user_message):
    client.beta.threads.messages.create(
        thread_id=thread.id, 
        role="user", 
        content=user_message
    )
    return client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )


def get_response(thread):
    return client.beta.threads.messages.list(thread_id=thread.id, order="asc")


def create_thread_and_run(user_input):
    thread = client.beta.threads.create()
    run = submit_message(ASSISTANT_ID, thread, user_input)
    return thread, run


# Pretty printing helper
def pretty_print(messages):
    #print("# Messages")
    # skip the first message, which is the user's question
    for m in messages:
        if m.role == "user":
            continue
        else:
            print(f"{m.role}: {m.content[0].text.value}\n")
    #print()


print("Welcome to the ASD Bot. Ask me a question about ASD. Type 'exit' to quit.\n")
thread = client.beta.threads.create()
# Start a loop to continuously prompt the user for input
while True:
    user_input = input("Ask the assistant: ")
    print()

    # Check if the user wants to exit the conversation
    if user_input.lower() in ['exit', 'quit']:
        break

    thread, run = create_thread_and_run(user_input)

    # # Create a message
    # message = client.beta.threads.messages.create(
    #     thread_id=thread.id,
    #     role="user",
    #     content=user_input
    # )
    
    # # Run the assistant
    # run = client.beta.threads.runs.create(
    #     thread_id=thread.id,
    #     assistant_id=ASSISTANT_ID
    #     # this will override the default prompt instructions we have given
    #     #instructions="Please address the user as Jane Doe. The user has a premium account."
    # )

    # # Step 5: Check the Run status
    # # By default, a Run goes into the queued state. You can periodically retrieve the Run to check on its status to see if it has moved to completed.
    # run = client.beta.threads.runs.retrieve(
    # thread_id=thread.id,
    # run_id=run.id
    # )

    # Wait for completion
    wait_on_run(run, thread)

    # # Retrieve all the messages added after our last user message
    # messages = client.beta.threads.messages.list(
    #     thread_id=thread.id, 
    #     order="asc", 
    #     after=message.id
    # )

    # Pretty print the messages
    pretty_print(get_response(thread))
