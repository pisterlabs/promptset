#!/usr/bin/env python3
"""USAGE: poetry run python3 ./talk.py "What is your question?"

tl;dr: It echoes back answers to questions, via text-to-speech.
"""
import datetime as DT, io, logging, os, sys  # noqa: E401
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from pydub import AudioSegment
from pydub.playback import play


class TalkDemo:
    """this demo doesn't leverage langchain, but rather newer OpenAI features

    Namely, the Text-To-Speech model and GPT-4 Turbo (released Nov 6th, 2023)
    also supports custom apps (GPTs, incl. via specialized "Assistant" APIs)
    that are NOT leveraged here. Instead, we do a simple chat query and speak
    the result in HD audio, which is saved in mp3 format to a folder on disk.
    """
    def __init__(self, system_prompt: str, **kwargs):
        # TODO: add options for non-default model usage
        self.client = OpenAI(**kwargs)  # temperature, etc.
        self.folder = Path(__file__).parent / "audio_files"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.system = system_prompt

    def answer(self, question: str, play=True, suffix_mp3="speech.mp3") -> str:
        gpt_model = os.getenv("GPT_MODEL_NAME", "gpt-4-1106-preview")
        # TODO: consider a better option (e.g. class props) over os.getenv above
        txt_system = self.system or "You are a helpful assistant.  Be very concise!"
        txt = question or "Isn't it a wonderful day to build something people love?"
        ans = self.client.chat.completions.create(
            model=gpt_model,
            messages=[
                {"role": "system", "content": txt_system},
                {"role": "user", "content": txt},
            ]
        )

        msg = ans.choices[0].message.content or "OK"
        res = self.client.audio.speech.create(
          input=msg,  # text to speech
          model="tts-1-hd",
          voice="echo",
        )

        now = DT.datetime.utcnow()
        mp3_path = self.folder / "_".join([now.isoformat(), suffix_mp3])
        res.stream_to_file(str(mp3_path))
        if play:
            self.play(mp3_path)
        return msg

    def play(self, mp3_path: Path) -> None:
        print("--- MP3 to play:", str(mp3_path))
        out = AudioSegment.from_file(str(mp3_path), format="mp3")
        out = out.set_frame_rate(44100)
        out = out.set_sample_width(2)
        out = out.set_channels(1)
        # TODO: neater decoding?

        wav = io.BytesIO()
        out.export(wav, format="wav")
        wav.seek(0)
        play(out)


def main(*args) -> None:
    if question := " ".join(args):
        print("---", DT.datetime.utcnow(), question)
        tts = TalkDemo(os.getenv("GPT_SYSTEM_PROMPT", ""))
        print("AI:", tts.answer(question))
    else:
        args = ("'What should I ask?'",)
        print("USAGE: poetry run python3", __file__, *args, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_dotenv(find_dotenv())
    main(*sys.argv[1:])
