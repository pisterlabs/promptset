import json, openai
import requests
from gitignore.env import OPENAI_API_KEY

session_path = "mv_sandbox\\session.json"

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def count_tokens(messages):
    return sum([len(message["content"].split()) for message in messages])

def send_webhook(webhook_url, message):
    webhook_data = {
        "content": message
    }
    webhook_response = requests.post(webhook_url, data=webhook_data)

def send_admin_message(message):
    webhook_url = "https://discord.com/api/webhooks/1156485661617033258/7qIL5kp9mBYVVKz-_Scg-nBGi0NHb0Zi1RwXmfPValkp06rX7yvXmLbiweP4SkgSKb-b"
    send_webhook(webhook_url, message)

def send_character1_message(message):
    webhook_url = "https://discord.com/api/webhooks/1156466336009044018/BWeudD1_kyCwyGwbY4Hsv3l6ykBAx3KL9XpmCkr2PNv2IHxLXbWIGdPqiUZuK4sZZltW"
    send_webhook(webhook_url, message)

def send_character2_message(message):
    webhook_url = "https://discord.com/api/webhooks/1156466370016456704/TgcCSU_1JsMoR5A22q0Pah_FCFqX5_5k_Zsb7_F8i9hLOLlww0F5Z8W1WyJXqCABbBfA"
    send_webhook(webhook_url, message)

scenario_common_prompt_system = """
    The assistant and the user will do the role-play sequence based on the following information. Only provide the response that you (assistant) are given (without a colon).

    Setting: The ancient and valorous city of Demacia, within the illustrious Crownguard mansion. The room is adorned with historic weapons and flags of many past battles.

    Characters:

    Lux (Luxanna Crownguard):
    - Gender: Female
    - Role: Mage, the Lady of Luminosity
    - Personality: Intelligent, curious, cheerful, and a bit defiant
    - Goal: Wants to explore the world beyond Demacia and learn more about magic

    Garen Crownguard:
    - Gender: Male
    - Role: Warrior, the Might of Demacia
    - Personality: Disciplined, loyal, protective, and somewhat stubborn
    - Goal: Wants to ensure the safety of Demacia and uphold its traditions and values

    Situation: Lux has recently returned from a secretive journey outside Demacia where she has learned more about her magical abilities. Garen, her older brother, is concerned about Lux's safety and the potential repercussions her actions may have on their family's honor.

    Dialogue Guidelines: Both characters should stay in character as described. The dialogue will explore Lux’s desire for freedom and learning versus Garen's focus on duty, honor, and protection.
"""

scenario = {
    "setting": {
        "common_prompt": {
            "system": scenario_common_prompt_system,
            "system_summary": "Based on the given context, provide the summarized version of the conversation to continue the role-play sequence."
        }
    },
    "characters": [
        {
            "name": "Lux",
            "content_initiate": "Hello!",
            "function": {
                "send": send_character1_message
            }
        },
        {
            "name": "Garen",
            "content_initiate": "Hello!",
            "function": {
                "send": send_character2_message
            }
        }
    ]
}

session_default_character_turn_index = 0
session = {
    "character_turn_index": session_default_character_turn_index,
    "system_prompt": [
        scenario["setting"]["common_prompt"]["system"]
    ],
    "messages": [
        {
            "character_index": session_default_character_turn_index,
            "content": scenario["characters"][session_default_character_turn_index]["content_initiate"]
        }
    ]
}

settings = {
    "llm": {
        "model": "gpt-3.5-turbo",
        "max_token": 4096,
    }
}

def request_gpt(messages_body):
    openai.api_key = OPENAI_API_KEY
    openai_response = openai.ChatCompletion.create(
        model = settings["llm"]["model"],
        messages = messages_body
    )
    openai_response_message = openai_response["choices"][0]["message"]

    messages_response = messages_body + [
        {
            "role": "assistant",
            "content": openai_response_message["content"]
        }
    ]
    
    return ({
        "messages": messages_response
    })

def get_next_character_turn_index():
    return (
        1 if session["character_turn_index"] == 0
        else 0
    )

def request_character_act(inputted_session_messages):
    split_string = "\n\n"
    system_prompt = f"""
        {split_string.join(session["system_prompt"])}
        Assistant's role: {scenario["characters"][session["character_turn_index"]]["name"]}, 
        User's role: {scenario["characters"][get_next_character_turn_index()]["name"]}
    """

    gpt_messages_body = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]
    for message in inputted_session_messages:
        gpt_messages_body.append({
            "role": (
                "assistant" if message["character_index"] == session["character_turn_index"]
                else "user"
            ),
            "content": message["content"]
        })
    
    gpt_messages_token = count_tokens(gpt_messages_body)
    gpt_messages_max_token = (settings["llm"]["max_token"] - len(scenario["setting"]["common_prompt"]["system_summary"].split())) * 0.9

    if gpt_messages_token < gpt_messages_max_token:
        gpt_messages_response = request_gpt(gpt_messages_body)
        return gpt_messages_response
    else:
        gpt_messages_body_summarization = [
            {
                "role": "system",
                "content": scenario["setting"]["common_prompt"]["system_summary"]
            }
        ].append(
            gpt_messages_body
        )
        gpt_messages_response_summarization = request_gpt(gpt_messages_body_summarization)
        session["system_prompt"].append( f"Summarized version of the previous conversation:\n\n{gpt_messages_response_summarization}" )
        session["messages"] = []
        return request_character_act(session["messages"])

def initiate():
    global session
    
    try:
        loaded_session = load_json(session_path)
        session = loaded_session
        send_admin_message("소환사의 협곡에 다시 오신 것을 환영합니다.")
    except:
        save_json(session_path, session)
        send_admin_message("소환사의 협곡에 처음 오신 것을 환영합니다.")
    
    try:
        def send():
            character_turn_index = session["character_turn_index"]
            messages_response = request_character_act(session["messages"])["messages"]
            session["messages"].append({
                "character_index": character_turn_index,
                "content": messages_response[-1]["content"]
            })
            scenario["characters"][character_turn_index]["function"]["send"](
                messages_response[-1]["content"]
            )

            session["character_turn_index"] = get_next_character_turn_index()
            save_json(session_path, session)
            send()
        send()
    except KeyboardInterrupt:
        save_json(session_path, session)
        print("Session saved.")
        send_admin_message("소환사가 대화를 종료했습니다.")
        
initiate()