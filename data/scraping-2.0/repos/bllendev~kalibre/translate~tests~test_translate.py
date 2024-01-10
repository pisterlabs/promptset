# django
from django.test import TestCase, SimpleTestCase
from django.conf import settings
from unittest.mock import patch

# local
from users.tests.factories import CustomUserFactory
from users.models import Email
from books.tests.factories import BookFactory
from translate._translate import EbookTranslate, Translate
from translate.constants import LANGUAGES
from books.models import Book
from books.tests.test_book_model import TEST_BOOK_PKL_PATH

# tools
import os
import json
import pickle
import openai
from googletrans import Translator


"""
docker compose exec web python manage.py test translate.tests.test_translate --noinput --parallel --failfast
"""


class TranslateConstants(SimpleTestCase):
    def test_langauge_constants(self):
        self.assertTrue(len(LANGUAGES) == 105)


class TranslateTest(SimpleTestCase):
    def setUp(self):
        self.test_translator = Translate()

    def test_init(self):
        # assert init true
        self.assertTrue(self.test_translator)

        # assert google api is default used
        self.assertTrue(self.test_translator.google_api)

    @patch("translate._translate.Translate._google_translate_text")
    def test_google_translate_used(self, mock_google_translate_text):
        text = self.test_translator.translate_text("hello", language="es")
        mock_google_translate_text.assert_called_once()

    def test_google_translate_correct_translation(self):
        text = self.test_translator.translate_text("hello", language="es")
        self.assertEquals(text.lower(), "hola")


class EbookTranslateTest(SimpleTestCase):
    _multiprocess_can_split_ = True
    _multiprocess_shared_ = False

    @classmethod
    def setUpClass(cls):
        super(EbookTranslateTest, cls).setUpClass()

        cls.book_file_path = TEST_BOOK_PKL_PATH

        # create book obj to test
        cls.test_book = BookFactory.build(test_book=True)  # orange tree book :)

        # open and assert test book
        with open(TEST_BOOK_PKL_PATH, "rb") as f:
            cls.test_epub = pickle.load(f)

    def test_epub_load(self):
        """make sure epub is loading in"""
        self.assertTrue(self.test_epub)

    # def test_get_book_translated(self):
    #     """
    #     testing book translation (should send to dev email)
    #     ... currently only testing english to spanish
    #     ... currently only testing google translate api
    #     """
    #     self.test_book.send(email_list=["bllendev@gmail.com"], language="es")
