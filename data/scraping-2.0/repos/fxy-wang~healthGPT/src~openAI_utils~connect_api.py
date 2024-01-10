from openai import OpenAI
import time
from decouple import config
from prompt_engineering import prompt1, prompt2, prompt3, prompt4, prompt5, prompt6
import os

OPENAI_KEY = os.environ.get('OPENAI_API_KEY')

# Function to initialize the OpenAI client
def initialize_openai_client():
    return OpenAI(api_key=OPENAI_KEY)


def files_upload(client):
    # Upload a file with an "assistants" purpose
    train_data_path = os.environ.get('TRAIN_PATH')
    file = client.files.create(
        file=open(train_data_path, "rb"),
        purpose='assistants'
    )
    return file

# Function to create an OpenAI assistant
def create_openai_assistant(client, file):
    assistant = client.beta.assistants.create(
        name="Enter your assistant name here",
        description=prompt1,
        model="gpt-4-1106-preview",
        tools=[{"type": "code_interpreter"}],#,{"type": "retrieval"}],
        file_ids=[]
        )
    return assistant

# Function to run the assistant and retrieve responses
def run_assistant_and_get_responses(client, assistant):
    thread = client.beta.threads.create()

    message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt1,
            file_ids=[]
        )
    
    message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt2,
            file_ids=[]
        )
    
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=prompt3,
        file_ids=[]
    )
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=prompt3,
        file_ids=[]
    )
    
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=prompt4,
        file_ids=[]
    )
    
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=prompt5,
        file_ids=[]
    )

    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=prompt6,
        file_ids=[]
    )   
    
    while True:
        print("User:")
        user_message = input()
        
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_message,
            #file_ids=[]
        )
    
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id,
            instructions=user_message
        )
        
        for _ in range(10):  # Retry up to 10 times
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            if run.status == "completed":
                messages = client.beta.threads.messages.list(
                    thread_id=thread.id
                )
                assistant_response = messages.data[0].content[0].text.value
                print()
                print("HealthHack:")
                print(assistant_response)
                print()
                break
            time.sleep(20)
        if run.status != "completed":
            break

# Main function to run the entire pipeline
def main():
    try:
        client = initialize_openai_client()
        file = files_upload(client)
        assistant = create_openai_assistant(client, file)
        #thread, message = create_thread_and_message(client, file)
        assistant_response = run_assistant_and_get_responses(client, assistant)
        #print(assistant_response)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()