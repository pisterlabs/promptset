import os
import openai
import uuid
import time
import datetime
from firebase_admin import firestore

model_name = "gpt-4-32k"
openai.api_key = os.getenv("OPENAI_KEY")
conv_length = 10


def chat(history: list) -> str:
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=history,
        presence_penalty=2.0,
    )

    return response.choices[0].message.content


def set_message(message: str) -> dict:
    return {"role": "user", "content": message}


def save_message(message: str) -> dict:
    return {"role": "assistant", "content": message}


def set_situation(prof1: str, prof2: str, name1: str, name2: str) -> tuple:
    with open("../prompts/conversation_situation_prompt.txt") as f:
        situation_text = f.read().format(prof1, prof2, name1, name2)
        situation_prompt = {"role": "system", "content": situation_text}

    topic = (
        openai.ChatCompletion.create(
            model=model_name,
            messages=[situation_prompt],
        )
        .choices[0]
        .message.content
    )

    with open("../prompts/conversation_setting_prompt.txt") as f:
        setting_text = f.read().format(topic, name1, name2)
        setting_prompt = {"role": "system", "content": setting_text}

    return setting_prompt, topic


def get_character_setting(animal_id: str) -> tuple:
    db = firestore.client()
    # firestoreから履歴を読み込む
    doc_ref = db.collection("characters").document(animal_id)
    character_data = doc_ref.get().to_dict()
    animal_name = character_data["name"]
    profile = character_data["profile"].replace("\\n", "\n")
    setting_text = character_data["history"][0]["content"].replace("\\n", "\n")
    setting_prompt = {"role": "system", "content": setting_text}

    return profile, animal_name, setting_prompt


def save_conversation(my_id: str, your_id: str, timestamp: datetime) -> tuple:
    conv_id = str(uuid.uuid4())
    db = firestore.client()
    doc_ref = db.collection("conversations").document(conv_id)
    
    doc_ref.set(
        {
            "history": [],
            "me": my_id,
            "you": your_id,
            "timestamp": timestamp,
        }
    )

    return conv_id


def save_history(history: list, conv_id: str):
    db = firestore.client()
    doc_ref = (
        db.collection("conversations")
        .document(conv_id)
    )

    doc_ref.update(
        {
            "history": [item for item in history if item["role"] != "system"],
        }
    )


def start_bot(animal_ids: list[str]):
    if len(animal_ids) != 2:
        return
    id1 = animal_ids[0]
    id2 = animal_ids[1]

    char1_prof, char1_name, prompt1 = get_character_setting(id1)
    char2_prof, char2_name, prompt2 = get_character_setting(id2)

    situation_prompt, topic = set_situation(
        char1_prof,
        char2_prof,
        char1_name,
        char2_name,
    )

    dt_now = datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=9)),
    )

    history1 = [prompt1]
    conv1 = save_conversation(id1, id2, dt_now)

    history2 = [prompt2]
    conv2 = save_conversation(id2, id1, dt_now)

    topic_prompt = [{"role": "system", "content": topic}]
    next_topic = {
        "role": "system",
        "content": "このシチュエーションの中で、関連する次の話題を考えてください。端的に3つ程度、ワードをください。",
    }

    for i in range(conv_length):
        if i % 2 == 0:
            if i == 0:
                history1.append(situation_prompt)
            res = chat(history1)
            history1.append(save_message(res))
            save_history(history1, conv1)
            history2.append(set_message(res))
            save_history(history2, conv2)
        else:
            res = chat(history2)
            history1.append(set_message(res))
            save_history(history1, conv1)
            history2.append(save_message(res))
            save_history(history2, conv2)

        topic_prompt.append(next_topic)
        res = chat(topic_prompt)
        event = {"role": "system", "content": res}
        topic_prompt.append({"role": "assistant", "content": res})
        history1.append(event)
        history2.append(event)

    save_dialy(id1, id2, history1)
    save_dialy(id2, id1, history2)


def save_dialy(animal_id: str, speaker_id: str, history: list):
    save_history = [chat for chat in history if chat["role"] != "system"]
    now = datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=9)),
    )
    memory_prompt = {
        "role": "system",
        "content": "会話の内容を日記小説のテイストで500文字程度にまとめてください。起承転結など、会話以外の内容も補完して記載してください。",
    }
    save_history.append(memory_prompt)

    memory = (
        openai.ChatCompletion.create(
            model=model_name,
            messages=save_history,
        )
        .choices[0]
        .message.content
    )

    db = firestore.client()
    doc_ref = db.collection("memories").document(f"{int(time.mktime(now.timetuple()))}")
    doc_ref.set(
        {
            "memory": memory,
            "animal_id": animal_id,
            "speaker_id": speaker_id,
            "timestamp": now,
        }
    )
