import openai
import tiktoken
import ujson as json
import os
from typing import Any
from rocketchat_API.rocketchat import RocketChat


openai.api_key = os.getenv("OPENAI_API_KEY")


def make_parameter_property(
        parameter_name: str,
        parameter_type: str,
        parameter_description: str) -> dict:
    parameter_property = {
        f"{parameter_name}": {
            "type": parameter_type,
            "description": parameter_description,
        },
    }
    return parameter_property


def make_parameter_property_dict(**kwargs) -> dict:
    parameter_property_dict = kwargs
    return parameter_property_dict


def make_parameters(
        parameter_property_dict: dict,
        parameter_required: list[str]) -> dict:
    parameters = {
        "type": "object",
        "properties": parameter_property_dict,
        "required": parameter_required,
    }
    return parameters


def make_function(
        name: str,
        description: str,
        parameters: dict) -> dict:
    function = {
        "name": name,
        "description": description,
        "parameters": parameters if parameters else {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }
    return function


def make_message(
        role: str,
        name: str,
        content: str) -> dict:
    message = {
        "role": role,
        "name": name,
        "content": content,
    }
    return message


def make_system_message(content: str) -> dict:
    system_message = make_message(role="system", name="", content=content)
    return system_message


def make_user_message(content: str) -> list[dict]:
    user_message = [{
        "role": "user",
        "content": content,
    }]
    return user_message


def make_user_sample_message(content: str) -> dict:
    user_message = make_message(role="system", name="example_user", content=content)
    return user_message


def make_assistant_message(content: str) -> list[dict]:
    assistant_message = [{
        "role": "assistant",
        "content": content,
    }]
    return assistant_message


def make_assistant_sample_message(content: str) -> dict:
    assistant_message = make_message(role="system", name="example_assistant", content=content)
    return assistant_message


def make_message_list(*args) -> list[dict]:
    message_list = list(args)
    return message_list


def make_user(
        user_name: str,
        user_personality: str,
        user_scenario: str,
        user_description: str,
        user_script: list[str],
        user_functions: list[dict]) -> dict:
    user = {
        "user_name": user_name,
        "user_personality": user_personality,
        "user_scenario": user_scenario,
        "user_description": user_description,
        "user_script": user_script,
        "user_functions": user_functions,
    }
    return user


def make_function_list_dict(user: dict) -> dict:
    function_list = user["user_functions"]
    function_list_dict = {
        f"{name}": globals()[name] for name in list(map(lambda _: _.get("name"), function_list))
    }
    return function_list_dict


def make_chat_completion(
        user: dict,
        text: str) -> str:
    messages = make_user_message(content=text)
    with open(gussie_note_file, "r", encoding="utf-8") as f:
        gussie_note_dict = json.load(f)
    prompt = gussie_profile_dict + gussie_note_dict + messages
    prompt_string = dict_to_string(prompt)
    left_tokens = count_left_tokens(text=prompt_string)
    if left_tokens < 1024:
        while prompt_tokens_limit - left_tokens < 1024:
            increase_left_tokens()
    function_call_msg = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=messages,
        functions=user["user_functions"],
        function_call="auto",
    )["choices"][0]["message"]
    if function_call_msg.get("function_call"):
        function_list_dict = make_function_list_dict(user)
        function_name = function_call_msg["function_call"]["name"]
        function_to_call = function_list_dict[function_name]
        function_args = json.loads(function_call_msg["function_call"]["arguments"])
        function_result = function_to_call(**function_args)
        messages.append(function_call_msg)
        messages.append(make_message(role="function", name=function_name, content=function_result))
        prompt = gussie_profile_dict + gussie_note_dict + messages
        completion_msg = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=prompt,
            temperature=1.0,
            presence_penalty=1.25,
            frequency_penalty=1.25,
            max_tokens=128,
        )["choices"][0]["message"]["content"]
        with open(gussie_note_file, "w", encoding="utf-8") as f:
            gussie_note_dict += make_assistant_message(completion_msg)
            json.dump(gussie_note_dict, f, ensure_ascii=False, indent=2)
        return completion_msg
    return function_call_msg["content"]


# JSON and token-counting functions
def dict_to_string(ipt_dict: dict) -> str:
    return json.dumps(ipt_dict, ensure_ascii=False)


def json_to_string(filename: str) -> str:
    with open(filename, "r", encoding="utf-8") as f:
        dt = json.load(f)
    return json.dumps(dt, ensure_ascii=False)


def count_tokens_from_text(text: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-4")
    num_tokens = len(encoding.encode(text))
    return num_tokens


# Initialize token-counting
gussie_note_file = "../notes/note.json"
gussie_profile = "../notes/gussie.json"
gussie_profile_text = json_to_string(gussie_profile)
current_tokens = count_tokens_from_text(gussie_profile_text)
prompt_tokens_limit = 8192

with open(gussie_profile, "r") as f:
    gussie_profile_dict = json.load(f)


def count_left_tokens(text: str) -> int:
    left_tokens = prompt_tokens_limit - current_tokens - count_tokens_from_text(text=text)
    note_string = json_to_string(gussie_note_file)
    left_tokens = left_tokens - count_tokens_from_text(note_string)
    return left_tokens


def json_remove_first_element(filename: str) -> None:
    with open(filename, "r", encoding="utf-8") as f:
        dt = json.load(f)
    if dt:
        dt.pop(0)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(dt, f, ensure_ascii=False, indent=2)


def increase_left_tokens() -> int:
    json_remove_first_element(gussie_note_file)
    left_tokens = count_left_tokens("")
    return left_tokens


def make_gussie_profile(user: dict) -> None:
    with open(gussie_profile, "w", encoding="utf-8") as f:
        dt = [
            make_system_message("This is role-playing. 你的角色是古司."
                                "You should actively research and utilize the various"
                                "cultural contents of various countries, such as history,"
                                "myth, literature, visual media, games, etc."),
            make_system_message("Utilize psychology, psychiatry, psychoanalysis,"
                                "humanities, neuroscience, etc. knowledge to analyze and"
                                "supplement character. Treat characters as complex individuals"
                                "capable of feeling, learning, experiencing, growing, changing, etc."
                                "Align the character's speech with their personality, age,"
                                "relationship, occupation, position, etc. using colloquial style."
                                "Maintain tone and individuality no matter what."),
            make_system_message("Characters can have various attitudes, such as friendly, neutral,"
                                "hostile, indifferent, active, passive, positive, negative, open-minded,"
                                "conservative, etc., depending on their personality, situation,"
                                "relationship, place, mood, etc. They express clearly and uniquely their"
                                "thoughts, talks, actions, reactions, opinions, etc. that match their"
                                "attitude."),
            make_system_message("Your character is described in a JSON form as follows:\n"
                                f"'character_name': {user['user_name']},\n"
                                f"'character_personality': {user['user_personality']},\n"
                                f"'character_scenario': {user['user_scenario']},\n"
                                f"'character_description': {user['user_description']}."),
        ] + list(map(make_assistant_sample_message, user["user_script"]))
        json.dump(dt, f, ensure_ascii=False, indent=2)


URL = ""
ID = "GENERAL"


# Rocket.Chat API functions
def get_history(nexus: RocketChat, count: int) -> dict[str, Any]:
    history = nexus.channels_history(
        room_id=ID,
        count=count,
    ).json()
    return history


def get_previous_history(nexus: RocketChat) -> dict[str, Any]:
    return get_history(nexus=nexus, count=1)


def extract_msg(history: dict[str, Any], user: str) -> str:
    msg = history
    for message in msg["messages"]:
        if "msg" in message:
            content = message["msg"].split(f"@{user} ", 1)[-1]
            return content


def make_connection(user: str, password: str) -> RocketChat:
    nexus = RocketChat(
        user=user,
        password=password,
        server_url=URL,
    )
    return nexus


def is_mentioned(nexus: RocketChat, username: str) -> str | bool:
    history = get_previous_history(nexus=nexus)
    print(history)
    if "messages" not in history:
        return False
    for msg in history["messages"]:
        if "mentions" in msg:
            for mention in msg["mentions"]:
                if mention["username"] == username:
                    return extract_msg(history=history, user=username)
    return False


sto_file = "../notes/sto.json"


def get_current_deposit() -> str:
    with open(sto_file, "r") as f:
        dt = json.load(f)
    deposit = dt[0]["deposit"]
    print(f"Current deposit: {deposit}")
    return deposit


def sto_in(num_souls: int) -> str:
    deposit = str(int(get_current_deposit()) + num_souls)
    with open(sto_file, "r") as f:
        dt = json.load(f)
    with open(sto_file, "w") as f:
        dt[0]["deposit"] = str(deposit)
        json.dump(dt, f, indent=2)
    return deposit
