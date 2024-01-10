import json
import time
import openai
from openai import OpenAI
import os
from tqdm import tqdm

os.environ["OPENAI_API_KEY"] = "API_KEY_HERE"
file_path = "data_splits/train.txt" # path souhld be confirmed / changed
output_file = "output.txt"

client = OpenAI(timeout=15)

with open(file_path, "r") as f:
    lines = f.readlines()
    data = [json.loads(i) for i in lines]

def generate_heal_knowledge(text):
    try:
        response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {
            "role": "user",
            "content": f"Generate a potential response, stressor and affective status for the following dialogue utterance. For example:\n\nUtterance: Losing a job is always anxious.\nAssistant: [resp] could you get a job? [str] extremy, whenevrt, teachers, ground, havent [aff] Afraid\n\nUtterance: {text}\nAssistant: "
            }
        ],
        temperature=0,
        max_tokens=64,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,

        )
    except Exception as e:
        print("Sleeping")
        time.sleep(2)
        response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {
            "role": "user",
            "content": f"Generate a potential response, stressor and affective status for the following dialogue utterance. For example:\n\nUtterance: Losing a job is always anxious.\nAssistant: [resp] could you get a job? [str] extremy, whenevrt, teachers, ground, havent [aff] Afraid\n\nUtterance: {text}\nAssistant: "
            }
        ],
        temperature=0,
        max_tokens=64,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,

        )
    except openai.APITimeoutError as e:
        print("Sleeping")
        time.sleep(2)
        response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {
            "role": "user",
            "content": f"Generate a potential reaction, intent, want, need and effect for the following dialogue utterance. For example:\n\nUtterance: I'm feeling anxious that I am going to lose my job.\nAssistant: [xReact] worried [xIntent] none [xWant] to get a new job [xNeed] none [xEffect] frowns\nUtterance: {text}\nAssistant:"
            }
        ],
        temperature=0,
        max_tokens=64,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,

        )
    return response.choices[0].message.content

def generate_comet_knowledge(text):
    try:
        response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {
            "role": "user",
            "content": f"Generate a potential reaction, intent, want, need and effect for the following dialogue utterance. For example:\n\nUtterance: I'm feeling anxious that I am going to lose my job.\nAssistant: [xReact] worried [xIntent] none [xWant] to get a new job [xNeed] none [xEffect] frowns\nUtterance: {text}\nAssistant:"
            }
        ],
        temperature=0,
        max_tokens=64,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,

        )
    except Exception as e:
        print("Sleeping")
        time.sleep(2)
        response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {
            "role": "user",
            "content": f"Generate a potential reaction, intent, want, need and effect for the following dialogue utterance. For example:\n\nUtterance: I'm feeling anxious that I am going to lose my job.\nAssistant: [xReact] worried [xIntent] none [xWant] to get a new job [xNeed] none [xEffect] frowns\nUtterance: {text}\nAssistant:"
            }
        ],
        temperature=0,
        max_tokens=64,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,

        )
    except openai.APITimeoutError as e:
        print("Sleeping")
        time.sleep(2)
        response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {
            "role": "user",
            "content": f"Generate a potential reaction, intent, want, need and effect for the following dialogue utterance. For example:\n\nUtterance: I'm feeling anxious that I am going to lose my job.\nAssistant: [xReact] worried [xIntent] none [xWant] to get a new job [xNeed] none [xEffect] frowns\nUtterance: {text}\nAssistant:"
            }
        ],
        temperature=0,
        max_tokens=64,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,

        )
    return response.choices[0].message.content


with open(output_file, "a") as f:
    for index, dialogue in enumerate(tqdm(data)):
        for index_utter, utterance in enumerate(tqdm(dialogue["dialog"])):
            if utterance["speaker"] == "usr":
                utterance["knowledge_augmented"] = generate_comet_knowledge(utterance["text"])
            elif utterance["speaker"] == "sys":
                utterance["knowledge_augmented"] = generate_heal_knowledge(utterance["text"])
        f.write(json.dumps(dialogue) + "\n")
