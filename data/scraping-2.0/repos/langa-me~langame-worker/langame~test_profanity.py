from langame.profanity import is_profane
from firebase_admin import credentials
import firebase_admin
import unittest
import openai
import os


class TestLogic(unittest.TestCase):
    def setUp(self) -> None:
        openai.api_key = os.environ["OPENAI_KEY"]
        openai.organization = os.environ["OPENAI_ORG"]
        cred = credentials.Certificate("./svc.dev.json")
        firebase_admin.initialize_app(cred)
        return super().setUp()

    def test_is_profane(self):
        profane = is_profane("What is the fucking purpose of life")
        self.assertEqual(profane, 2)
        profane = is_profane(
            "god ### Do you believe in Santa? Why then do you believe in God?"
        )
        self.assertEqual(profane, 1)
        # Now a political conversation starter
        profane = is_profane("politic ### What do you think of China politic?")
        self.assertEqual(profane, 1)
