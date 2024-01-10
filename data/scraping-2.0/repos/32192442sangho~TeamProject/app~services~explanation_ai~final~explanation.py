import base64

import openai
import json
import os
from gtts import gTTS
from app.services.explanation_ai.primary_model.distance import distance
from app.services.explanation_ai.primary_model.favor_unfavor import favor
from app.services.explanation_ai.primary_model.if_use_spell_well import if_use_spell_well
from app.services.explanation_ai.primary_model.spell1 import spell1
from app.services.explanation_ai.primary_model.spell2 import spell2
from app.services.explanation_ai.primary_model.what_to_what import what_to_what
from moviepy.editor import VideoFileClip


def final(VIDEO_PATH):
    video = VIDEO_PATH.split("/")[3].split(".")[0]
    GIF_PATH = f"app/services/explanation_ai/gif/{video}.gif"

    try:
        os.remove(GIF_PATH)
    except OSError:
        pass


    # Set up the OpenAI API client
    openai.api_key = "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"

    b_proto1 = spell1(VIDEO_PATH)
    b_proto2 = spell2(VIDEO_PATH)
    a_proto = favor(VIDEO_PATH)

    if b_proto1 == "don't use" and b_proto2 == "don't use":
        b = "0"
    elif b_proto1 == "use" and b_proto2 == "don't use":
        b = "1"
    elif b_proto1 == "don't use" and b_proto2 == "use":
        b = "1"
    else:
        b = "2"

    a = f"팀의 파워는 {a_proto}"
    b = f"플레이어는 {b}개의 주문을 사용했다."
    c = f"플레이어는 적과의 거리조절을 {distance(VIDEO_PATH)}"
    d = f"전투는 {what_to_what(VIDEO_PATH)} 싸움이다."
    e = f"{if_use_spell_well(VIDEO_PATH)}"


    if b_proto1 == "don't have" and b_proto2 == "don't have":
        e = "스펠이 없었다."

    # Define the list of sentences that describe the situation
    situation = [
        "플레이어는 리그 오브 레전드를 플레이 중이다.",
        "플레이어는 사망했다",
        a,
        b,
        c,
        d,
        e
    ]
    # Concatenate the sentences together to form a single string
    input_text = " ".join(situation)
    print(input_text)
    # Generate a sentence that describes the situation using the GPT model
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=input_text,
        max_tokens=300
    )

    # Extract the generated sentence from the API response
    output_text = response.choices[0].text.strip()
    print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@{output_text}")
    tts = gTTS(text=output_text, lang='ko')
    audio_output_file = f"app/services/explanation_ai/audio/{video}.mp3"
    if os.path.exists(audio_output_file):
        os.remove(audio_output_file)
    tts.save(audio_output_file)
    with open(audio_output_file, "rb") as audio_file:
        audio_data = audio_file.read()

    a = VideoFileClip(VIDEO_PATH).resize(height=240).set_fps(5)
    a.write_gif(GIF_PATH)
    with open(GIF_PATH, "rb") as gif_file:
        gif_base64 = base64.b64encode(gif_file.read()).decode('utf-8')

    output = json.dumps({"output_text": output_text, "tts": audio_data.decode('latin1'), "gif": gif_base64})

    print(output)
    return output

