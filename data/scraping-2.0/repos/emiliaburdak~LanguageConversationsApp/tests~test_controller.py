import json
import unittest
from flask_testing import TestCase
from werkzeug.security import generate_password_hash
from unittest.mock import patch
from app import db
from app.models import User, Conversation, Message
from main import app
from app.service import ChatAPIError


class ControllerTests(TestCase):

    def create_app(self):
        app.config["TESTING"] = True
        app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///test.db"  # Use an in-memory database for testing
        return app

    def setUp(self):
        db.session.remove()
        db.drop_all()
        db.create_all()
        hashed_password = generate_password_hash("testpassword", method="sha256")
        self.test_user = User(
            username="testuser",
            name="Test User",
            password=hashed_password
        )
        db.session.add(self.test_user)
        db.session.commit()

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test_login_required(self):
        login = self.client.post("login", json=dict(username="testuser", password="testpassword"))
        login_data = json.loads(login.data.decode("utf-8"))
        bearer_token = login_data["token"]
        return bearer_token

    def test_home(self):
        bearer_token = self.test_login_required()
        home_response = self.client.get("home", headers={"Authorization": f"Bearer {bearer_token}"})
        decoded_home_response = json.loads(home_response.data.decode("utf-8"))
        self.assert200(home_response)
        self.assertEqual(decoded_home_response["message"], "Welcome to home!")
        self.assertEqual(decoded_home_response["username"], "testuser")

    def test_create_conversation(self):
        bearer_token = self.test_login_required()
        input_data = {"language": "spanish", "conversation_name": "conv123"}
        conv_response = self.client.post("/conversation", headers={"Authorization": f"Bearer {bearer_token}"},
                                         json=input_data)
        decoded_conv_response = json.loads(conv_response.data.decode("utf-8"))

        self.assert200(conv_response)
        self.assertEqual(decoded_conv_response["name"], "conv123")
        self.assertIsNotNone(decoded_conv_response["id"])

    def test_get_conversations_empty(self):
        bearer_token = self.test_login_required()
        convs_response = self.client.get("/conversations", headers={"Authorization": f"Bearer {bearer_token}"})
        decoded_convs_response = json.loads(convs_response.data.decode("utf-8"))
        self.assert200(convs_response)
        self.assertEqual(decoded_convs_response, [])

    def test_get_conversations(self):
        bearer_token = self.test_login_required()
        user2 = User(username="testuser2", name="Test User2",
                     password=(generate_password_hash("testpassword2", method="sha256")))
        db.session.add(user2)
        db.session.commit()
        conversation1 = Conversation(conversation_name="Test Conversation 1", user_id=self.test_user.id,
                                     language="Spanish")
        conversation2 = Conversation(conversation_name="Test Conversation 2", user_id=user2.id,
                                     language="Spanish")
        db.session.add(conversation1)
        db.session.add(conversation2)
        db.session.commit()
        convs_response = self.client.get("/conversations", headers={"Authorization": f"Bearer {bearer_token}"})
        decoded_convs_response = json.loads(convs_response.data.decode("utf-8"))
        self.assert200(convs_response)
        self.assertEqual(decoded_convs_response[0]["name"], "Test Conversation 1")
        self.assertEqual(len(decoded_convs_response), 1)

    def _create_examples_to_db(self):
        bearer_token = self.test_login_required()

        user2 = User(username="testuser2", name="Test User2",
                     password=(generate_password_hash("testpassword2", method="sha256")))
        db.session.add(user2)
        db.session.commit()
        conversation3 = Conversation(conversation_name="Test Conversation 3", user_id=user2.id,
                                     language="English")
        conversation1 = Conversation(conversation_name="Test Conversation 1", user_id=self.test_user.id,
                                     language="Spanish")
        conversation2 = Conversation(conversation_name="Test Conversation 2", user_id=self.test_user.id,
                                     language="Spanish")
        db.session.add(conversation1)
        db.session.add(conversation2)
        db.session.add(conversation3)
        db.session.commit()
        message1 = Message(message_text="Hello", conversation_id=conversation1.id, is_user=True)
        message2 = Message(message_text="Hi there", conversation_id=conversation1.id, is_user=False)
        db.session.add(message1)
        db.session.add(message2)
        db.session.commit()

        return bearer_token

    def test_get_conversation(self):
        bearer_token = self._create_examples_to_db()

        conv_response = self.client.get(f"/conversation/1",
                                        headers={"Authorization": f"Bearer {bearer_token}"})
        decode_conv_response = json.loads(conv_response.data.decode("utf-8"))

        self.assert200(conv_response)
        self.assertEqual(decode_conv_response["id"], "1")
        self.assertEqual(decode_conv_response["conversation_name"], "Test Conversation 1")
        self.assertEqual(decode_conv_response["language"], "Spanish")
        self.assertTrue("beginning_date" in decode_conv_response)
        self.assertTrue("last_message_date" in decode_conv_response)
        self.assertEqual(len(decode_conv_response["messages"]), 2)
        self.assertEqual(decode_conv_response["messages"][0]["message_text"], "Hello")
        self.assertEqual(decode_conv_response["messages"][0]["id"], 1)
        self.assertTrue(decode_conv_response["messages"][0]["is_user"])
        self.assertTrue("timestamp" in decode_conv_response["messages"][0])

    def _mock_response(self, test_answer_summary):
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": f"{test_answer_summary}"
                    }
                }
            ]
        }
        return mock_response

    def _prepare_and_call_get_chat_response(self, test_answer_summary, user_message="test_message"):
        bearer_token = self.test_login_required()
        conversation = Conversation(conversation_name="Test conversation", user_id=self.test_user.id,
                                    language='Spanish')
        db.session.add(conversation)
        db.session.commit()
        mock_response = self._mock_response(test_answer_summary)

        with patch("openai.ChatCompletion.create", return_value=mock_response) as mock_openai_call:
            input_data = {"TTS_message": user_message}

            chat_response = self.client.post(f"/response/{conversation.id}",
                                             headers={"Authorization": f"Bearer {bearer_token}"},
                                             json=input_data)

        return mock_openai_call, chat_response, bearer_token, conversation

    def test_get_chat_response(self):
        test_answer_summary = '{"answer": "Test response from OpenAI", "summary": "Testing"}'
        mock_openai_call, chat_response, bearer_token, conversation = self._prepare_and_call_get_chat_response(
            test_answer_summary)

        decoded_chat_response = json.loads(chat_response.data.decode("utf-8"))
        mock_openai_call.assert_called_once()
        self.assert200(chat_response)
        self.assertEqual(decoded_chat_response["chat_message"], "Test response from OpenAI")

    def test_save_chat_response_to_db(self):
        test_answer_summary = '{"answer": "Test response from OpenAI", "summary": "Testing"}'
        mock_openai_call, chat_response, bearer_token, conversation = self._prepare_and_call_get_chat_response(
            test_answer_summary)

        db_response = self.client.get(f"/conversation/{conversation.id}",
                                      headers={"Authorization": f"Bearer {bearer_token}"})
        decode_db_response = json.loads(db_response.data.decode("utf-8"))
        self.assertEqual(decode_db_response["messages"][0]["message_text"], "test_message")
        self.assertEqual(decode_db_response["messages"][1]["message_text"], "Test response from OpenAI")

    def test_prep_empty_response(self):
        test_answer_summary = '{"answer": "", "summary": ""}'
        self._test_save_invalid_response(test_answer_summary)

    def test_prep_json_response(self):
        test_answer_summary = '{"invalid_json_response"}'
        self._test_save_invalid_response(test_answer_summary)

    def _test_save_invalid_response(self, test_answer_summary):
        mock_openai_call, chat_response, bearer_token, conversation = self._prepare_and_call_get_chat_response(
            test_answer_summary)
        db_response = self.client.get(f"/conversation/{conversation.id}",
                                      headers={"Authorization": f"Bearer {bearer_token}"})
        decode_db_response = json.loads(db_response.data.decode("utf-8"))
        self.assertEqual(len(decode_db_response["messages"]), 1)
        self.assertEqual(decode_db_response["messages"][0]["message_text"], "test_message")

    def test_chat_empty_response(self):
        test_answer_summary = '{"answer": "", "summary": ""}'
        mock_openai_call, chat_response, bearer_token, conversation = self._prepare_and_call_get_chat_response(
            test_answer_summary)
        decoded_chat_response = json.loads(chat_response.data.decode("utf-8"))
        mock_openai_call.assert_called_once()
        self.assert200(chat_response)
        self.assertEqual(decoded_chat_response["chat_message"], "I have technical problem with answer, please repeat")

    def test_chat_invalid_json_response(self):
        test_answer_summary = '{"invalid_json_response"}'
        mock_openai_call, chat_response, bearer_token, conversation = self._prepare_and_call_get_chat_response(
            test_answer_summary)
        decoded_chat_response = json.loads(chat_response.data.decode("utf-8"))
        mock_openai_call.assert_called_once()
        self.assert200(chat_response)
        self.assertEqual(decoded_chat_response["chat_message"], "I have technical problem with answer, please repeat")

    def test_invalid_user_message(self):
        test_answer_summary = '{"answer": "Test response from OpenAI", "summary": "Testing"}'
        _, json_error, _, _ = self._prepare_and_call_get_chat_response(test_answer_summary, "")
        json_data = json.loads(json_error.data)
        self.assertEqual(json_data["error"], "I have technical problem with answer, please repeat")

    def test_get_hint(self):
        bearer_token = self._create_examples_to_db()
        test_answer_summary = "This is your hint"
        mock_response = self._mock_response(test_answer_summary)

        with patch("openai.ChatCompletion.create", return_value=mock_response) as mock_openai_call:
            guidance_response = self.client.post(f"/hint/1", headers={"Authorization": f"Bearer {bearer_token}"})
            decode_guidance_response = json.loads(guidance_response.data.decode("utf-8"))

            mock_openai_call.assert_called_once()
            self.assertEqual(decode_guidance_response["guidance_response"], "This is your hint")

    def test_none_input_get_advanced_version(self):
        bearer_token = self._create_examples_to_db()
        test_answer_summary = "This is your advanced version"
        mock_response = self._mock_response(test_answer_summary)

        with patch("openai.ChatCompletion.create", return_value=mock_response):
            advanced_version_response = self.client.post(f"/advanced_version/1",
                                                         headers={"Authorization": f"Bearer {bearer_token}"})
            decode_advanced_version_response = json.loads(advanced_version_response.data.decode("utf-8"))

            self.assertEqual(decode_advanced_version_response["error"],
                             "There is no sentence to correct, please use hint instead")

    def test_get_advanced_version(self):
        bearer_token = self._create_examples_to_db()
        test_answer_summary = "This is your advanced version"
        mock_response = self._mock_response(test_answer_summary)

        with patch("openai.ChatCompletion.create", return_value=mock_response) as mock_openai_call:
            input_data = {"chat_message": "chat_message"}
            advanced_version_response = self.client.post(f"/advanced_version/1",
                                                         headers={"Authorization": f"Bearer {bearer_token}"},
                                                         json=input_data)
            decode_advanced_version_response = json.loads(advanced_version_response.data.decode("utf-8"))

            mock_openai_call.assert_called_once()
            self.assertEqual(decode_advanced_version_response["guidance_response"], "This is your advanced version")

    def test_invalid_input_get_hint(self):
        bearer_token = self.test_login_required()
        conversation1 = Conversation(conversation_name="Test Conversation 1", user_id=self.test_user.id,
                                     language="Spanish")
        db.session.add(conversation1)
        test_answer_summary = "This is your hint"
        mock_response = self._mock_response(test_answer_summary)

        with patch("openai.ChatCompletion.create", return_value=mock_response):
            hint_response = self.client.post(f"/hint/1", headers={"Authorization": f"Bearer {bearer_token}"})
            decode_hint_response = json.loads(hint_response.data.decode("utf-8"))

            self.assertEqual(decode_hint_response["error"],
                             "Please, start conversation before using hint or sentence advanced correction")

    def test_invalid_input_advanced_version(self):
        bearer_token = self.test_login_required()
        conversation1 = Conversation(conversation_name="Test Conversation 1", user_id=self.test_user.id,
                                     language="Spanish")
        db.session.add(conversation1)
        test_answer_summary = "This is your hint"
        mock_response = self._mock_response(test_answer_summary)

        with patch("openai.ChatCompletion.create", return_value=mock_response):
            advanced_version_response = self.client.post(f"/advanced_version/1",
                                                         headers={"Authorization": f"Bearer {bearer_token}"})
            decode_advanced_version_response = json.loads(advanced_version_response.data.decode("utf-8"))

            self.assertEqual(decode_advanced_version_response["error"],
                             "Please, start conversation before using hint or sentence advanced correction")

    def test_chat_api_error_get_hint(self):
        bearer_token = self._create_examples_to_db()
        with patch("openai.ChatCompletion.create", side_effect=ChatAPIError("Failed to get a response from the chat")):
            hint_response = self.client.post(f"/hint/1", headers={"Authorization": f"Bearer {bearer_token}"})
            decode_hint_response = json.loads(hint_response.data.decode("utf-8"))

            self.assertEqual(decode_hint_response["error"], "Failed to get a response from the chat")

    def test_chat_api_error_get_advanced_version(self):
        bearer_token = self._create_examples_to_db()
        with patch("openai.ChatCompletion.create", side_effect=ChatAPIError("Failed to get a response from the chat")):
            input_data = {"chat_message": "chat_message"}
            advanced_version_response = self.client.post(f"/advanced_version/1",
                                                         headers={"Authorization": f"Bearer {bearer_token}"},
                                                         json=input_data)
            decode_advanced_version_response = json.loads(advanced_version_response.data.decode("utf-8"))

            self.assertEqual(decode_advanced_version_response["error"], "Failed to get a response from the chat")

    def _get_translation(self, payload_to_translation):
        bearer_token = self.test_login_required()
        translation_response = self.client.post(f"/translation",
                                                headers={"Authorization": f"Bearer {bearer_token}"},
                                                json=payload_to_translation)
        decoded_translation_response = json.loads(translation_response.data.decode("utf-8"))
        return decoded_translation_response

    def test_correct_payload_to_translation(self):
        payload_to_translation = {"word_to_translate": "computadora",
                                  "sentence_to_translate": "me gusta usar mi computadora", "source_lang": "ES",
                                  "target_lang": "EN-GB"}
        decoded_translation_response = self._get_translation(payload_to_translation)
        self.assertEqual(decoded_translation_response["translated_sentence"], "I like to use my computer")
        self.assertEqual(decoded_translation_response["translated_word"], "computer")

    def test_typo_in_payload_to_translation(self):
        payload_to_translation = {"word_to_translate": "compatadora",
                                  "sentence_to_translate": "me gusta usar mi compatadora", "source_lang": "ES",
                                  "target_lang": "EN-GB"}
        decoded_translation_response = self._get_translation(payload_to_translation)
        self.assertEqual(decoded_translation_response["translated_sentence"], "I like to use my computer")
        self.assertEqual(decoded_translation_response["translated_word"], "compatadora")

    def test_invalid_payload_to_translation(self):
        payload_to_translation = {"word_to_translate": "computadora",
                                  "source_lang": "ES",
                                  "target_lang": "EN-GB"}
        decoded_translation_response = self._get_translation(payload_to_translation)
        self.assertEqual(decoded_translation_response["error"],
                         "Incorrect data format. Make sure you press the word and try again.")

    def _add_to_dictionary(self, payload_to_dictionary):
        bearer_token = self.test_login_required()
        translation_response = self.client.post(f"/dictionary",
                                                headers={"Authorization": f"Bearer {bearer_token}"},
                                                json=payload_to_dictionary)
        decoded_translation_response = json.loads(translation_response.data.decode("utf-8"))
        return decoded_translation_response

    def test_correct_payload_add_to_dictionary(self):
        payload_to_dictionary = {
            "word_to_dictionary": "computadora",
            "contex_sentence": "me gusta usar mi computadora",
            "source_lang": "ES",
            "target_lang": "EN-GB"
        }
        decoded_translation_response = self._add_to_dictionary(payload_to_dictionary)
        self.assertEqual(decoded_translation_response["translated_word"], "computer")
        self.assertEqual(decoded_translation_response["translated_contex_sentence"], "I like to use my computer")

    def test_typo_payload_add_to_dictionary(self):
        payload_to_dictionary = {
            "word_to_dictionary": "compatadora",
            "contex_sentence": "me gusta usar mi computadora",
            "source_lang": "ES",
            "target_lang": "EN-GB"
        }
        decoded_translation_response = self._add_to_dictionary(payload_to_dictionary)
        self.assertEqual(decoded_translation_response["translated_word"], "compatadora")
        self.assertEqual(decoded_translation_response["translated_contex_sentence"], "I like to use my computer")

    def test_invalid_payload_add_to_dictionary(self):
        payload_to_dictionary = {
            "word_to_dictionary": "computadora",
            "contex_sentence": "me gusta usar mi computadora",
            "target_lang": "EN-GB"
        }
        decoded_translation_response = self._add_to_dictionary(payload_to_dictionary)
        self.assertEqual(decoded_translation_response["error"],
                         "Incorrect data format. Make sure you provide the word and try again.")


if __name__ == "__main__":
    unittest.main()
