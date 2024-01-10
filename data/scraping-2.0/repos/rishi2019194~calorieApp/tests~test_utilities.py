import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import sys
import os
import openai

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to the Python path
sys.path.append(parent_dir)
from utilities import *


class TestUtilities(unittest.TestCase):

    @patch('openai.ChatCompletion.create')
    def test_chatcompletion_pass(self, mock_openai_create):
        # Mock the OpenAI API call
        mock_openai_create.return_value = {
            'choices': [{
                'message': {
                    'content': 'Mocked response'
                }
            }]
        }

        user_input = "Hello"
        impersonated_role = "Assistant"
        explicit_input = "Provide assistance"
        chat_history = "User: Hi\nAssistant: Hello"
        result = chatcompletion(user_input, impersonated_role, explicit_input,
                                chat_history)
        self.assertEqual(result, 'Mocked response')

    @patch('openai.ChatCompletion.create')
    def test_chatcompletion_fail(self, mock_openai_create):
        # Mock the OpenAI API call
        mock_openai_create.return_value = {
            'choices': [{
                'message': {
                    'content': 'Mocked response'
                }
            }]
        }

        user_input = "Hello"
        impersonated_role = "Fitness coach"
        explicit_input = "Provide fitness advice"
        chat_history = "User: Hi\nAssistant: Hello"

        result = chatcompletion(user_input, impersonated_role, explicit_input,
                                chat_history)

        self.assertNotEqual(result, '')

    @patch('openai.ChatCompletion.create')
    def test_chat_pass(self, mock_openai_create):
        # Mock the OpenAI API call
        mock_openai_create.return_value = {
            'choices': [{
                'message': {
                    'content': 'Mocked response'
                }
            }]
        }

        chat_history = "User: Hi\nAssistant: Hello"
        name = "Assistant"
        chatgpt_output = ""
        user_input = "How are you?"
        history_file = "test_history.txt"
        impersonated_role = "Assistant"
        explicit_input = "Provide assistance"

        result = chat(chat_history, name, chatgpt_output, user_input,
                      history_file, impersonated_role, explicit_input)

        self.assertEqual(result, 'Mocked response')

    @patch('openai.ChatCompletion.create')
    def test_chat_fail(self, mock_openai_create):
        # Mock the OpenAI API call
        mock_openai_create.return_value = {
            'choices': [{
                'message': {
                    'content': 'Mocked response'
                }
            }]
        }

        chat_history = "User: Hi\nAssistant: Hello"
        name = "Assistant"
        chatgpt_output = ""
        user_input = "How are you?"
        history_file = "test_history.txt"
        impersonated_role = "Assistant"
        explicit_input = "Provide assistance"

        result = chat(chat_history, name, chatgpt_output, user_input,
                      history_file, impersonated_role, explicit_input)

        self.assertNotEqual(result, 'Incorrect response')

    def test_get_response_pass(self):
        # Mocking the chat function
        with patch('utilities.chat', return_value='Mocked response'):
            chat_history = "User: Hi\nAssistant: Hello"
            name = "Assistant"
            chatgpt_output = ""
            user_text = "How are you?"
            history_file = "test_history.txt"
            impersonated_role = "Assistant"
            explicit_input = "Provide assistance"

            result = get_response(chat_history, name, chatgpt_output,
                                  user_text, history_file, impersonated_role,
                                  explicit_input)

            self.assertEqual(result, 'Mocked response')

    @patch('utilities.chat', return_value='Mocked response')
    def test_get_response_fail(self, mock_chat):
        chat_history = "User: Hi\nAssistant: Hello"
        name = "Assistant"
        chatgpt_output = ""
        user_text = "How are you?"
        history_file = "test_history.txt"
        impersonated_role = "Assistant"
        explicit_input = "Provide assistance"

        result = get_response(chat_history, name, chatgpt_output, user_text,
                              history_file, impersonated_role, explicit_input)

        self.assertNotEqual(result, 'Incorrect response')

    def test_get_entries_for_email_pass(self):

        mock_db = MagicMock()
        entries_data = [{'email': 'test@example.com', 'date': '2023-11-23'}]
        mock_db.calories.find.return_value = entries_data

        email = 'test@example.com'
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)

        result, [] = get_entries_for_email(mock_db, email, start_date,
                                           end_date)

        self.assertEqual(result, entries_data)

    def test_get_entries_for_email_fail(self):

        mock_db = MagicMock()
        entries_data = [{'email': 'test@example.com', 'date': '2023-11-23'}]
        mock_db.calories.find.return_value = entries_data

        email = 'test@example.com'
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)

        result, _ = get_entries_for_email(mock_db, email, start_date, end_date)

        self.assertNotEqual(result, [{
            'email': 'wrong@example.com',
            'date': '2023-11-23'
        }])

    def test_calc_bmi_pass(self):
        result = calc_bmi(70, 175)
        # Match correct value
        self.assertEqual(result, 22.86)

    def test_calc_bmi_fail(self):
        result = calc_bmi(70, 175)
        # Match correct value
        self.assertNotEqual(result, 12)

    def test_get_bmi_category_pass(self):
        result_underweight = get_bmi_category(18.0)
        result_normal = get_bmi_category(22.0)
        result_overweight = get_bmi_category(27.0)
        result_obese = get_bmi_category(30.0)

        self.assertEqual(result_underweight, 'Underweight')
        self.assertEqual(result_normal, 'Normal Weight')
        self.assertEqual(result_overweight, 'Overweight')
        self.assertEqual(result_obese, 'Obese')

    def test_get_bmi_category_fail(self):
        result_underweight = get_bmi_category(18.0)
        result_normal = get_bmi_category(22.0)
        result_overweight = get_bmi_category(27.0)
        result_obese = get_bmi_category(30.0)

        self.assertNotEqual(result_underweight, 'Normal Weight')
        self.assertNotEqual(result_normal, 'Obese')
        self.assertNotEqual(result_overweight, 'Underweight')
        self.assertNotEqual(result_obese, 'Overweight')

    def test_total_calories_to_burn_pass(self):
        result = total_calories_to_burn(70, 60)
        # Expected result: (70 - 60) * 7700 = 10000
        self.assertEqual(result, 77000)

    def test_total_calories_to_burn_fail(self):
        result = total_calories_to_burn(70, 60)
        # This should fail, as the expected result is 10000
        self.assertNotEqual(result, 5000)

    def test_calories_to_burn_pass(self):
        target_date = datetime.today() + timedelta(days=10)
        start_date = datetime.today() - timedelta(days=5)
        result = calories_to_burn(2000, 1500, target_date, start_date)
        self.assertEqual(result, 1166)

    def test_calories_to_burn_fail(self):
        target_date = datetime.today() + timedelta(days=10)
        start_date = datetime.today() - timedelta(days=5)
        result = calories_to_burn(2000, 1500, target_date, start_date)
        # This should fail, as the expected result is 100
        self.assertNotEqual(result, 50)


if __name__ == '__main__':
    unittest.main()
