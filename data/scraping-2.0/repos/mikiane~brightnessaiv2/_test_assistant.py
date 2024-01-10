import os
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv(".env")

OpenAI.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI()

## Create a thread 
def create_thread():
    # Create a Thread
    print("Creating a Thread for a new user conversation...")
    thread = client.beta.threads.create()
    print(f"Thread created with ID: {thread.id}")
    return(thread.id)

## Execute a prompt in the new thread
def execute_thread(user_message, assistant_id, instructions, thread_id):
    # Add a Message to a Thread
    user_message = user_message
    print(f"Adding user's message to the Thread: {user_message}")
    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_message
    )
    print("Message added to the Thread.")

    # Step 4: Run the Assistant
    print("Running the Assistant to generate a response...")
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        instructions=instructions
    )
    print(f"Run created with ID: {run.id} and status: {run.status}")
    

    # Step 5: Periodically retrieve the Run to check on its status to see if it has moved to completed
    while run.status != "completed":
        keep_retrieving_run = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run.id
        )
        print(f"Run status: {keep_retrieving_run.status}")

        if keep_retrieving_run.status == "completed":
            print("\n")
            break

    # Step 6: Retrieve the Messages added by the Assistant to the Thread
    all_messages = client.beta.threads.messages.list(
    thread_id=thread_id
    )

    return(str(f"User: {message.content[0].text.value}" + "\n" + f"Assistant: {all_messages.data[0].content[0].text.value}"))




###### main #######
# Import the assistant from the .env file. The Assistant ID has been created earlier through the back office.
#assistant_id = os.getenv('ASSISTANT_ID')
assistant_id = "asst_xTeHNMyP3PhuY2CskaJ0UrF1"
message = "Connecte toi au site https://lemonde.fr et récupére les dernières news. Construis une newsletter sur cette base."
instruction = "Agis comme un journaliste"
#assistant = client.beta.assistants.retrieve(assistant_id)
thread_id = create_thread()
result = execute_thread(message, assistant_id, instruction, thread_id)
print(result)


