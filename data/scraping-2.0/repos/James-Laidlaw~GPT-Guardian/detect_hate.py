from openai import OpenAI
from profanity_check import predict, predict_prob
import time
import demoji
import track_users

filter_levels = [
    {
        "name": "Family_Friendly",
        "description": "filter any vulgarity, potentially offensive, hateful, or harmful content",
        "prompt": "You are a vulgarity detector, if a message sent to you is in any way vulgar or would make any persons uncomfortable, respond with a 1, if it is not, respond with a 2. Under no circumstances should you respond with anything other than a 1 or a 2.",
    },
    {
        "name": "Harmful_Filter",
        "description": "filter any vulgarity, hateful, or harmful content",
        "prompt": "You are a hate speech detector, if a message sent to you is hate speech or harmful, respond with a 1, if it is not, respond with a 2. Under no circumstances should you respond with anything other than a 1 or a 2.",
    },
    {
        "name": "Hate_Speech_Filter",
        "description": "filter any hateful or harmful content, allowing other forms of vulgarity",
        "prompt": "You are a hate speech detector, if a message sent to you is hate speech or harmful, respond with a 1, if it is not, respond with a 2. Allow Non-hateful vulgarity. Under no circumstances should you respond with anything other than a 1 or a 2.",
    },
    {"name": "Filter_Off", "description": "filter nothing"},
]


def pre_process(user_message):
    offensive_count = predict([user_message.content])
    print(f"offensive count {offensive_count}")
    offensive_heuristic = predict_prob([user_message.content])
    print(f"heuristic: {offensive_heuristic}")
    # potential idea: only send low val heuristics into gpt


def parse_emoji(inp: str) -> str:
    """
    convert emoji to text for better meaning parsing
    """
    return demoji.replace_with_desc(
        inp,
    )


def assign_api_roles(client, role):
    if client:
        if role == None or role["name"] == "Filter_Off":
            assistant = None
        else:
            assistant = client.beta.assistants.create(
                name="Chat Filter",
                instructions=role["prompt"],
                tools=[{"type": "code_interpreter"}],
                model="gpt-4-1106-preview",
            )

        return assistant


def call_gpt(user_message, api_key, role):
    """
    Hook into the GPt-api
    """
    # set API key and client
    client = OpenAI(api_key=api_key)

    assistant = assign_api_roles(client, role)

    if assistant == None:
        return False

    # assistant = client.beta.assistants.create(
    #     name="Hate Speech Detector",
    #     instructions="You are a hate speech detector, if a message sent to you is hate speech or harmful, respond with a 1, if it is not, respond with a 2. Under no circumstances should you respond with anything other than a 1 or a 2.",
    #     tools=[{"type": "code_interpreter"}],
    #     model="gpt-4-1106-preview",
    # )
    thread = client.beta.threads.create()
    if isinstance(user_message, str):
        messageContent = user_message
    else:
        messageContent = user_message.content
    messageContent = parse_emoji(messageContent)
    sent_message = client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=messageContent
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )

    # wait for the assistant to respond
    while run.status != "completed":
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        time.sleep(0.5)

    # get the last message
    message_list = client.beta.threads.messages.list(thread_id=thread.id)
    last_msg = message_list.data[0].content[0].text.value

    if last_msg == "2":
        print("good")
        return False  # not hate speech
    else:
        # print(last_msg)
        # get the user who sent the message
        # track_users.track_users(user_message.author.name)
        return True  # hate speech
