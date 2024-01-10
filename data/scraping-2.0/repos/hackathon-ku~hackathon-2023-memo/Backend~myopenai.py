from openai import OpenAI
import dotenv
import json
import time
import os
dotenv.load_dotenv()

client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY"),
)

def get_Topic(user_input):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "user", "content": f"Generate chat topic from {user_input} \n make sure to return only topic name"},
        ]
    )
    return completion.choices[0].message.content

def get_thread(thread_id):
    return client.beta.threads.retrieve(thread_id=thread_id)

def create_thread_and_run(user_input):
    thread = client.beta.threads.create()
    run = submit_message(os.getenv("ASSISTANT_ID"), thread, user_input)
    return thread, run

def submit_message(assistant_id, thread, user_message):
    client.beta.threads.messages.create(
        thread_id=thread.id, 
        role="user", 
        content=user_message,
        file_ids=[os.getenv("FILE_ID")],
    )
    return client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )

def get_response(thread):
    return client.beta.threads.messages.list(thread_id=thread.id, order="asc")

def get_GPT_response(thread):
    # Resmessages = []
    messages = client.beta.threads.messages.list(thread_id=thread.id, order='desc')
    for m in messages:
        if m.role == "assistant":
            return m.content[0].text.value
    
def pretty_print(messages):
    print("# Messages")
    for m in messages:
        print(f"{m.role}: {m.content[0].text.value}")
    print()

def wait_on_run(run, thread):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run

# thread1, run2 = create_thread_and_run(
#     "What I should enroll in year 3 semester 2"
# )
# run2 = wait_on_run(run2, thread1)
# print(get_GPT_response(thread1))

# print("=========================================")

# run2 = submit_message(os.getenv("ASSISTANT_ID"), thread1, "how about year 2 semester 1?")
# run2 = wait_on_run(run2, thread1)
# print(get_GPT_response(thread1))

# print("=========================================")


# run2 = submit_message(os.getenv("ASSISTANT_ID"), thread1, "THX")
# run2 = wait_on_run(run2, thread1)
# print(get_GPT_response(thread1))
