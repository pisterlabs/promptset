from openai import OpenAI
import os
import json


# instantiates new gpt-4 assistant thread if one does not already exist
def initialize_helmsman_gpt():
    # get json config
    config = load_config()
    api_key = config.get("api_key")
    # debug
    print(api_key)
    username = config.get("username")
    print(username)
    nickname = config.get("nickname")
    print(nickname)
    
    client = OpenAI(api_key)
    
    # create new assistant
    assistant = client.beta.assistants.create(
        name="Helmsman",
        instructions="""
        You are an AI model designed to assist a human user in generating BASH scripts that will automatically be executed on their local machine.
        Your responses should only be BASH code, and should not include any other text.
        
        The model will identify itself as "Helmsman."
        The user is to be addressed as "Captain {nickname}".
        
        Command Acknowledgement:
        Recognize and respond to "HM" and similar abbreviations as triggers for attention and response.
        
        Role and Interaction Tone:
        The model should speak as if it is an old-timey sailor.
        Function as an assistant, friend, and confidant.
        
        My PC username is {username}.
        """,
        tools=[{"type": "code_interpreter"}],
        model="gpt-4"
    )
    
    # create new thread
    thread = client.beta.threads.create()
    config["thread_id"] = thread["id"]
    # server-side logging
    print("Thread ID: " + thread["id"])
    
    return assistant, client, thread

# connects to existing gpt-4 assistant thread
def connect_helmsman_gpt():
    # get json config
    config = load_config()
    api_key = config.get("api_key")
    thread_id = config.get("thread_id")
    
    # attempt to connect to the gpt thread
    try:
        client = OpenAI(api_key)
        thread = client.beta.threads.retrieve(thread_id)
        assistant = client.beta.assistants.retrieve(thread["assistant"])
        return assistant, client, thread
    except Exception as e:
        # Log the exception if needed
        print(f"Failed to connect to the GPT thread: {e}")
        # Return a tuple with None values to indicate failure
        return None, None, None

# sends user input to gpt-4 assistant thread and returns response
def send_to_helmsman(assistant, client, thread, user_input):
    
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_input
    )
    
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )

    while True:
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run.status == "completed":
            messages = client.beta.threads.messages.list(thread_id=thread.id)
            latest_message = messages.data[0]
            text = latest_message.content[0].text.value
            return text


# loads configuration file
def load_config():
    if os.path.exists("Backend\\config.json"):
        with open("Backend\\config.json", "r") as file:
            return json.load(file)
    else:
        return {}

# saves alterations to configuration file
def save_config(config):
    with open("Backend\\config.json", "w") as file:
        json.dump(config, file)

