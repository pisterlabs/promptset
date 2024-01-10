import os
from unittest import TestCase

import openai
from dotenv import load_dotenv
from gtts import gTTS

from speech.utils import mp3_to_text


class TestAudio(TestCase):
    def setUp(self) -> None:
        load_dotenv()
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")

    def test_transcribe(self):
        transcript = mp3_to_text(os.path.join(self.data_dir, "123.mp3"))
        self.assertEqual(transcript, 'Раз, два, три.')

    def test_tts(self):
        tts = gTTS("""Привет, меня зовут Лиила.""", lang='ru')
        tts.save('hello.mp3')
        self.assertTrue(os.path.exists('hello.mp3'))
        os.remove('hello.mp3')
