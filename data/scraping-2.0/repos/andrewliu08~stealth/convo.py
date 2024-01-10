import json
import logging
import os
import requests
import subprocess
import time

# TODO: add logging for latencies

import boto3
import openai

from secret_keys import (
    AWS_ACCESS_KEY,
    AWS_SECRET_ACCESS_KEY,
    DEEPL_API_KEY,
    OPENAI_API_KEY,
)
from stream_sample import convo


starter_prompt = '''You are a fluent {0} speaker. You are having a conversation in {0} with the goal: "{1}".
Provide {2} options for how you might start the conversation with the other person. For each option, provide an English translation of what you said. Follow the format in the triple quotes:
"""
Option 1:
"Response 1 in {0}"
"Translation of response 1 in English"

Option 2:
"Response 2 in {0}"
"Translation of response 2 in English"
...
"""
'''

history_prompt = '''You are a fluent {0} speaker. You are having a conversation in {0} with the goal: "{1}". The conversation so far is in triple quotes:
"""
{2}
"""
Provide {3} options for what you might say to the other person. For each option, provide an English translation of what you said. Follow the format in the triple quotes:
"""
Option 1:
"Response 1 in {0}"
"Translation of response 1 in English"

Option 2:
"Response 2 in {0}"
"Translation of response 2 in English"
...
"""
'''

AWS_POLLY_LANG_TO_VOICE = {
    "English": "Matthew",
    "French": "Lea",
    "Mandarin": "Zhiyu",
}

DEEPL_LANG_TO_CODE = {
    "English": "EN-US",
    "French": "FR",
    "Mandarin": "ZH",
}


def get_conversation_string(conversation_history):
    conversation_str = ""
    for i, turn in enumerate(conversation_history):
        if i % 2 == 0:
            conversation_str += f"You: {turn}\n"
        else:
            conversation_str += f"Other: {turn}\n"
    return conversation_str


def get_prompt(language, intent, conversation_history, num_response_options):
    """
    conversation_history is a list of strings in the respondent language,
    each string representing a turn in the conversation.
    """
    if len(conversation_history) == 0:
        prompt = starter_prompt.format(language, intent, num_response_options)
        return prompt
    else:
        conversation_str = get_conversation_string(conversation_history)
        prompt = history_prompt.format(
            language, intent, conversation_str, num_response_options
        )
        return prompt


def gpt_responses(prompt, stream):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt},
        ],
        stream=stream,
    )
    return response


def parse_options(response_content):
    """
    Assumes response_content is formatted as:
    Option 1:
    "Response 1 in respondent language"
    "Translation of response 1 in English"

    Option 2:
    "Response 2 in respondent language"
    "Translation of response 2 in English"

    ...
    """
    options = response_content.split("Option")
    result = []
    for option in options:
        if len(option.strip()) == 0:
            continue
        lines = [line.strip() for line in option.split("\n") if line.strip()]
        result.append((lines[1], lines[2]))

    return result


def deepl_translate(text, source_language_code, target_language_code):
    url = "https://api-free.deepl.com/v2/translate"
    headers = {
        "Authorization": "DeepL-Auth-Key {}".format(DEEPL_API_KEY),
    }
    data = {
        "text": text,
        "source_lang": source_language_code,
        "target_lang": target_language_code,
    }

    response = requests.post(url, headers=headers, data=data)
    return response


def polly_tts(text, voice_id, engine, output_file, polly_client=None):
    """
    voice_id is used to select the language and voice.
    engine can either be "standard" or "neural"
    """
    if polly_client is None:
        # Initialize a session using Amazon Polly
        polly_client = boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name="us-west-2",
        ).client("polly")

    response = polly_client.synthesize_speech(
        VoiceId=voice_id, OutputFormat="mp3", Text=text, Engine=engine
    )

    # Save the audio to a file
    with open(output_file, "wb") as file:
        file.write(response["AudioStream"].read())


def play_audio(audio_file):
    subprocess.run(["afplay", audio_file])


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(current_dir, "log_convo.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s: %(message)s",
    )

    openai.api_key = OPENAI_API_KEY
    polly_client = polly_client = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="us-west-2",
    ).client("polly")
    polly_engine = "standard"

    initiator_language = "English"
    respondent_language = input("Select language: ")
    intent = input("What is the goal of the conversation? ")
    inititiator_lang_voice = AWS_POLLY_LANG_TO_VOICE[initiator_language]
    respondent_lang_voice = AWS_POLLY_LANG_TO_VOICE[respondent_language]

    # Contains conversation history in respondent langauge
    convo_history_respondent_lang = []
    convo_history_initiator_lang = []
    while True:
        print()
        # Initiator text
        prompt = get_prompt(
            respondent_language,
            intent,
            convo_history_respondent_lang,
            num_response_options=3,
        )
        logging.info(f"GPT prompt:\n{prompt}")
        # response = convo
        response = gpt_responses(prompt, stream=True)
        deltas = []
        for chunk in response:
            choice = chunk["choices"][0]
            if choice["finish_reason"] == "stop":
                continue
            else:
                print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
                deltas.append(chunk["choices"][0]["delta"]["content"])
        logging.info("OPENAI response:\n{}".format(deltas))
        print("\n")

        response_options = parse_options("".join(deltas))
        option_choice = int(input("Option choice (1-3): "))
        if option_choice < 1 or option_choice > 3:
            break
        response_choice, response_translation = response_options[option_choice - 1]
        convo_history_respondent_lang.append(response_choice)
        convo_history_initiator_lang.append(response_translation)
        print("\nYou:", response_choice)
        print("You translation:", response_translation)
        logging.info("Option choice: {}".format(option_choice))
        logging.info("You: {}".format(response_choice))
        logging.info("You translation: {}".format(response_translation))

        # Generate initiator's audio and play it
        initiator_audio_path = os.path.join(
            current_dir,
            "audio_history/{}.mp3".format(len(convo_history_respondent_lang)),
        )
        polly_tts(
            response_choice,
            respondent_lang_voice,
            polly_engine,
            initiator_audio_path,
            polly_client,
        )
        play_audio(initiator_audio_path)

        # Respondent text
        respondent_text = input("\nEnter respondent's text: ")
        convo_history_respondent_lang.append('"' + respondent_text + '"')

        respondent_translation = deepl_translate(
            respondent_text,
            DEEPL_LANG_TO_CODE[respondent_language],
            DEEPL_LANG_TO_CODE[initiator_language],
        )
        respondent_translation = respondent_translation.json()["translations"][0][
            "text"
        ]
        convo_history_initiator_lang.append('"' + respondent_translation + '"')
        print("Respondent translation:", respondent_translation)

        # Generate respondent's audio and play it
        respondent_audio_path = os.path.join(
            current_dir,
            "audio_history/{}.mp3".format(len(convo_history_respondent_lang)),
        )
        polly_tts(
            respondent_translation,
            inititiator_lang_voice,
            polly_engine,
            respondent_audio_path,
            polly_client,
        )
        play_audio(respondent_audio_path)

    print()
    print("-" * 30)
    print("Conversation history:")
    time.sleep(0.5)
    for i, turn in enumerate(convo_history_respondent_lang):
        audio_file_path = os.path.join(
            current_dir, "audio_history/{}.mp3".format(i + 1)
        )
        if i % 2 == 0:
            print("You:", turn)
            print("You translation:", convo_history_initiator_lang[i])
            play_audio(audio_file_path)
        else:
            print("Other:", turn)
            print("Other translation:", convo_history_initiator_lang[i])
            play_audio(audio_file_path)
    print("-" * 30)


if __name__ == "__main__":
    main()
