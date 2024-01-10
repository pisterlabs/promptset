import re
import json
import openai

def translation(org_dialogues, lang, openai_token):
    prompt = f"""Translate the following conversations to "{lang}" while maintaining relevant context and ensuring that the output duration matches the start_time and end_time. Add special characters like .... or ---- where necessary based on the context. Provide the output in JSON code format for easy copying and pasting. My goal is to use this text for TTS (Text-to-Speech) purposes. Please make sure the translations are accurate and fluent.
    Input format : 
    [[dialogue, start_time, end_time],[dialogue, start_time, end_time],..]
    Output format = [dialogue, dialgoue,..]
    Don't include time stamps in output"""

    preprocessed_dialogues = [
        (dialogue["text"], dialogue["start_time"], dialogue["end_time"])
        for dialogue in org_dialogues
    ]

    preprocessed_dialogues_str = json.dumps(preprocessed_dialogues)

    prompt += preprocessed_dialogues_str

    openai.api_key = openai_token

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=2200
    )

    translated_text = response["choices"][0]["text"]
    # print(translated_text)
    dialogue_lines = re.findall(r'"([^"]*)"', translated_text)
    # print(dialogue_lines)

    translated_dialogues = org_dialogues.copy()
    
    for i in range(len(translated_dialogues)):
        translated_dialogues[i]["text"] = dialogue_lines[i]

    return translated_dialogues