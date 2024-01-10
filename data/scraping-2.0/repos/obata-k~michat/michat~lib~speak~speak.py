import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from enum import Enum
import json

import openai
from dotenv import load_dotenv
from playsound import playsound
from voicevox_core import AccelerationMode, VoicevoxCore

open_jtalk_dict_dir = "./open_jtalk_dic_utf_8-1.11"
acceleration_mode = AccelerationMode.AUTO
system_root = Path("system")


class ChatGPTFeature(Enum):
    ZUNDAMON = "ずんだもん"
    ENE = "エネ"
    MIKU = "初音ミク"

    @classmethod
    def get_names(cls) -> list:
        return [i.name for i in cls]

    @classmethod
    def get_values(cls) -> list:
        return [i.value for i in cls]

    @classmethod
    def index(cls, value) -> int:
        return cls.get_values().index(value)

    @classmethod
    def value_of(cls, target_value):
        for e in ChatGPTFeature:
            if e.value == target_value:
                return e
        raise ValueError("{} is not a valid feature".format(target_value))


class ChatGPT:
    def __init__(self, max_token_size):
        self.__max_token_size = max_token_size
        dotenv_path = Path(os.path.join(os.getcwd(), ".env"))
        if dotenv_path.exists():
            load_dotenv(dotenv_path)
        openai.api_key = os.environ.get("OPENAI_API_KEY")

    @property
    def max_token_size(self):
        return self.__max_token_size

    @max_token_size.setter
    def max_token_size(self, n):
        self.__max_token_size = n

    # history は（DBやファイルなど）外部で保持している
    def generate(self, system_text, user_text, history=None):
        messages = []
        if history is None:
            history = []
        for h in history:
            messages.append(h)
        messages.extend(
            [
                {
                    "role": "system",
                    "content": system_text,
                },
                {
                    "role": "user",
                    "content": user_text,
                },
            ]
        )
        # GPT-3でテキストを生成する
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=int(self.max_token_size),
            n=1,
            stop=None,
            temperature=0.5,
        )

        # GPT-3の生成したテキストを取得する
        text = response.choices[0].message.content.strip()
        history = history + [
            {
                "role": "user",
                "content": user_text,
            },
            {"role": "assistant", "content": text},
        ]
        return (text, history)


class ChatGPTWithEmotion(ChatGPT):
    def __init__(self, max_token_size):
        super().__init__(max_token_size)
        self.system_emotion = system_root / Path("system-emotion.txt")

    def trim_and_parse(self, text):
        lines = []
        payload = None
        for line in text.splitlines():
            try:
                payload = json.loads(line)
                continue
            except ValueError:
                pass
            if "感情パラメータ" not in line and line != "":
                lines.append(line)
        return "\n".join(lines), payload

    def generate(self, system_text, user_text, history=None):
        with open(self.system_emotion, "r") as f:
            system_text += f.read()
        generated, new_history = super().generate(system_text, user_text, history)
        response, params = self.trim_and_parse(generated)
        return (response, new_history, params)


class Audio:
    def __init__(self, speaker_id):
        self.speaker_id = speaker_id

    # voicevoxでテキストを音声に変換する
    def transform(self, text):
        self.core = VoicevoxCore(
            acceleration_mode=acceleration_mode, open_jtalk_dict_dir=open_jtalk_dict_dir
        )
        self.core.load_model(self.speaker_id)
        self.audio_query = self.core.audio_query(text, self.speaker_id)

    # 音声をファイルに保存する
    def save_wav(self, out):
        out.write_bytes(self.wav)

    def get_wav(self):
        self.wav = self.core.synthesis(self.audio_query, self.speaker_id)
        return self.wav

    # 音声を再生する
    def play(self, file):
        playsound(file)


def setup_log(log_file, log_level):
    FORMAT = "%(asctime)s: [%(levelname)s] %(message)s"
    logging.basicConfig(format=FORMAT)

    level_num = getattr(logging, log_level.upper(), None)
    if not isinstance(level_num, int):
        raise ValueError("Invalid log level: %s" % log_level)
    logger = logging.getLogger(__name__)
    logger.setLevel(level_num)

    if log_file == "stdout":
        stdout_handler = logging.StreamHandler()
        logger.addHandler(stdout_handler)
    else:
        file_handler = logging.FileHandler(filename=log_file)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def system_text(feature=ChatGPTFeature.ZUNDAMON):
    if feature is ChatGPTFeature.ZUNDAMON:
        system = Path("system-zundamon.txt")
    elif feature is ChatGPTFeature.ENE:
        system = Path("system-ene.txt")
    elif feature is ChatGPTFeature.MIKU:
        system = Path("system-miku.txt")
    else:
        raise ValueError("invalid ChatGPT feature was set")

    with open(system_root / system, "r") as f:
        text = f.read()
    return text
