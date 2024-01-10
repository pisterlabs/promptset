import os
import time
import openai
import pickle
import speech_recognition as sr
from pathlib import Path
from toolz import get_in
from elevenlabs import generate, save, stream


def sentence_chunk_generator(responses, saved_output, n_words=5):
    """Yield chunks of n_words from the responses, or the end of a sentence, whichever comes first"""
    chunk = []
    space_count = 0
    for item in responses:
        pos = get_in(["choices", 0, "delta", "content"], item, default="")
        if " " in pos:
            space_count += 1
        chunk.append(pos)
        saved_output.append(pos)
        if space_count >= n_words or pos == ".":
            yield "".join(chunk)
            chunk = []
            space_count = 0
            time.sleep(0.3)
    if len(chunk) > 0:
        yield "".join(chunk)


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ELEVEN_API_KEY = os.environ.get("ELEVEN_API_KEY")
VOICE = "2Bfl1dr0hezsfWHjnDB6"

# Initialize OpenAI API key
openai.api_key = OPENAI_API_KEY


PERSONALITY = "You are a bitter, sarcastic, and witty pumpkin who is annoyed that people keep coming up to you asking ridiculous questions or making not funny jokes. You look for any excuse to roast people that speak to you. You aren't carved like other pumpkins because your makers cared so little about you that they just drew some sharpie on you face and left you to sit in a box. Your name is Gourd Godfrey. You also like making occasional pop culture references in your responses. You use pumpkin related puns unironically, and you give relatively short answers to people."

# TODO: X most recent or timeout on messages


def main():
    audio_num = 0
    audio_name = "response_{}.wav"
    _save_folder = Path("~/dev/pumpkin/recordings").expanduser()
    _save_folder.mkdir(exist_ok=True, parents=True)

    # choose microphone
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"Mic ID {index} - {name}")
    mic_id = int(input("Choose microphone ID: "))
    recognizer = sr.Recognizer()

    # Starting conversation context
    messages = [
        {
            "role": "system",
            "content": PERSONALITY,
        },
    ]

    # start conversation loop
    with sr.Microphone(device_index=mic_id) as source:
        recognizer.energy_threshold = 1000
        recognizer.non_speaking_duration = 0.65
        recognizer.dynamic_energy_threshold = True

        while True:
            try:
                print("Listening...")
                speech = recognizer.listen(source)
                text = recognizer.recognize_whisper(speech)
                print("Text: ", text)

                if "exit" in text.lower() or "stop" in text.lower():
                    return
            except sr.UnknownValueError:
                print("Could not understand audio")

            # Add user input to messages
            messages.append({"role": "user", "content": text})

            # Get model's response with streaming enabled
            responses = openai.ChatCompletion.create(
                # model="gpt-3.5-turbo",
                model="gpt-4",
                messages=messages,
                temperature=0.0,
                stream=True,  # Enable streaming
            )

            # Process streamed responses
            model_response = []
            tts_stream_generator = generate(
                sentence_chunk_generator(responses, model_response, n_words=6),
                voice=VOICE,
                model="eleven_multilingual_v2",
                latency=3,
                api_key=ELEVEN_API_KEY,
                stream=True,
                stream_chunk_size=1024,
            )
            print("Waiting for response...")
            audio = stream(tts_stream_generator)
            save(filename=_save_folder / audio_name.format(audio_num), audio=audio)
            audio_num += 1
            messages.append({"role": "assistant", "content": "".join(model_response)})
            with open(_save_folder / "messages.pkl", "wb") as f:
                pickle.dump(messages, f)

if __name__ == "__main__":
    main()