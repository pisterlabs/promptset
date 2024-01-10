from openai import OpenAI
from dotenv import load_dotenv
import os
import time

load_dotenv()


ASSISTANT_ID = os.environ['ASSISTANT_ID']


client = OpenAI()

def wait_on_run(run, thread):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run


def main():
    
    assistant = client.beta.assistants.retrieve(assistant_id=ASSISTANT_ID)
    
    thread = client.beta.threads.create()
    print(thread)
    
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content="How well can you play SC2?",
    )
    
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )
    
    run = wait_on_run(run, thread)
    
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    
    print(messages)



if __name__ == '__main__': 
    main()