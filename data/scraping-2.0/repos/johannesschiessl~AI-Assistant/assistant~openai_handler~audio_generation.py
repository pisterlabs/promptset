from openai import OpenAI
import datetime

def get_current_date_time():
    current_date_time = datetime.datetime.now()
    formatted_date_time = current_date_time.strftime("%Y-%m-%d%H:%M:%S.%f")[:-3]
    return formatted_date_time

def generate_audio(prompt):
    client = OpenAI()
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=prompt,
    )

    response.stream_to_file("data\\ai_assistant_tts.mp3")
    return "data\\ai_assistant_tts.mp3"