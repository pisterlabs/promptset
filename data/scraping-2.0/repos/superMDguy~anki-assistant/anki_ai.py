import os
import hashlib
import time
import threading
import subprocess
import logging
import sys

import numpy as np
from faster_whisper import WhisperModel
from anki.collection import Collection
from openai import OpenAI
from html2text import html2text
import webview
import pyaudio
import torch

from config import OPENAI_KEY, ANKI_PATH

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
logging.basicConfig()


TEST = len(sys.argv) > 1 and sys.argv[1] == "noaudio"

# Audio settings
SAMPLE_RATE = 16000
CHUNK = int(SAMPLE_RATE / 10)
num_samples = 1536  # Number of samples to use for the VAD model
audio = pyaudio.PyAudio()


client = OpenAI(api_key=OPENAI_KEY)


if not TEST:
    # Takes about 0.5 seconds, small.en is about 1.5s on my machine
    model = WhisperModel(
        "tiny.en",
        device="cpu",
        compute_type="float32",
    )

    # For dialog detection
    vad_model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad"
    )


def confidence(chunk):
    """
    Use Silero VAD to detect if the user is speaking.
    """
    audio_int16 = np.frombuffer(chunk, np.int16)
    abs_max = np.abs(audio_int16).max()
    audio_float32 = audio_int16.astype("float32")
    if abs_max > 0:
        audio_float32 *= 1 / 32768
    audio_float32 = audio_float32.squeeze()
    return vad_model(torch.from_numpy(audio_float32), SAMPLE_RATE).item()


def transcribe(audio_data):
    """
    Use Whisper to transcribe audio.
    """
    audio_data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
    segments, _ = model.transcribe(
        audio_data / np.max(audio_data),
        language="en",
        beam_size=5,
        without_timestamps=True,
        initial_prompt="Z_i = A X squared plus B X plus C",
    )
    return "".join(x.text for x in segments)


def transcribe_answer():
    """
    Stream audio from user's microphone and transcribe.

    Listening Algorithm:
    - Continously listen for audio chunks once the user starts speaking
    - If the user stops speaking for 0.8 seconds, transcribe the phrase
    - Once the user hasn't spoken for 2 seconds, finalize the transcription.
    """
    if TEST:
        return input("Your text: ")

    # Record audio until no talking for 0.8 seconds
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    log.debug("Listening")

    prev_confidence = []
    data = []
    transcription = ""

    stop = threading.Event()

    last_spoken = [time.time()]

    def threaded_listen():
        while not stop.is_set():
            audio_chunk = stream.read(num_samples)
            chunk_confidence = confidence(audio_chunk)
            prev_confidence.append(chunk_confidence)

            mid_phrase = np.sum(prev_confidence[-5:]) > 5 * 0.7
            currently_speaking = chunk_confidence > 0.75

            if mid_phrase or currently_speaking:
                data.append(audio_chunk)

            if currently_speaking:
                last_spoken[0] = time.time()

    threading.Thread(target=threaded_listen, daemon=True).start()

    while not len(data):
        # Wait for user to start talking
        time.sleep(0.1)

    while True:
        speaking_gap = time.time() - last_spoken[0]

        if speaking_gap < 0.8:
            time.sleep(0.8 - speaking_gap)
        elif speaking_gap < 2.0 and len(data):
            log.debug(f"start transcribe {speaking_gap}, {len(data)}")
            stt_start = time.time()
            next_chunk = b"".join(data)
            data.clear()
            transcription += transcribe(next_chunk)
            log.debug(f" stt {time.time() - stt_start}")
        else:  # speaking_gap > 2.0
            log.info("return")
            assert len(data) == 0
            stop.set()
            log.debug(transcription)
            return transcription

    stop.set()


def tts(text):
    """
    Play text as audio, using cache when possible.
    """
    if TEST:
        return log.debug(text)
    key = hashlib.sha1(text.encode("utf-8")).hexdigest()
    if os.path.exists(f"cached_audio/{key}.mp3"):
        # Playing at 80% speed works better
        subprocess.call(["afplay", f"cached_audio/{key}.mp3", "-r", "0.8"])
    else:
        subprocess.call(["say", make_latex_speakable(text)])

        # Use OpenAI TTS to cache audio for next time
        def cache():
            client.audio.speech.create(
                model="tts-1", voice="nova", input=text
            ).stream_to_file(f"cached_audio/{key}.mp3")

        threading.Thread(target=cache, daemon=True).start()


def make_latex_speakable(text):
    if "\\" not in text and "$" not in text:
        return text
    return (
        client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {
                    "role": "user",
                    "content": "Translate all Latex/symbols and say this out loud:\n"
                    + text,
                },
            ],
            temperature=0,
        )
        .choices[0]
        .message.content
    )


def main_backend(window):
    def display_html(html):
        window.evaluate_js(f"window.updateHtml(String.raw`{html}`);")

    collection = Collection(ANKI_PATH)

    try:
        while current_card := collection.sched.getCard():
            # TODO: handle cloze cards
            if "basic" not in current_card.note_type()["name"].lower():
                log.debug("Skipping cloze")
                collection.sched.bury_cards([current_card.id])
                continue
            (question, answer) = current_card.note().fields
            answer = html2text(answer)

            if "latex" in answer.lower() or "img" in answer.lower():
                log.debug("Skipping rendered latex/image")
                collection.sched.bury_cards([current_card.id])
                continue

            display_html(current_card.render_output(browser=True).question_and_style())
            tts(question)

            current_card.timer_started = time.time()  # Start timer for scoring
            user_response = transcribe_answer()

            if "skip card" in user_response.lower():
                collection.sched.bury_cards([current_card.id])
                continue

            if "i don't know" in user_response.lower():
                score = 1
            else:
                score = int(
                    client.chat.completions.create(
                        model="gpt-3.5-turbo-1106",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a PhD in applied mathematics, giving flashcards to a student. Rate how correct the student was, on a scale of 1-4:\n"
                                + "1 - Doesn't know the answer. Totally incorrect, blank, or gibberish.\n"
                                + "2 - Shows some knowledge.\n"
                                + "3 - Partially incorrect.\n"
                                + "4 - Demonstrates understanding. Responses that lack specific details can still get a 4. Responses that explain the concept in a different way can still get a 4.\n"
                                + "Answer only numerically!",
                            },
                            {
                                "role": "user",
                                "content": f'Q: {question}\nA:{answer}\Student Response: "{user_response}"',
                            },
                        ],
                        temperature=0,
                        max_tokens=1,
                    )
                    .choices[0]
                    .message.content
                )

            # Flash screen red or green depending on score
            window.evaluate_js(
                f"window.flashScreen('{'#CC020255' if score < 4 else '#02CC0255'}');"
            )
            log.info(f"Score: {score}")
            collection.sched.answerCard(current_card, score)

            if score < 4:
                # Show correct answer if user got it wrong
                display_html(
                    current_card.render_output(browser=True).answer_and_style()
                )
                tts(answer)
                time.sleep(3)
    finally:
        collection.close()


if __name__ == "__main__":
    window = webview.create_window(
        "Anki Voice Assistant",
        html=open("display_card.html", "r").read(),
    )
    webview.start(main_backend, window)
