import pytest
import pathlib
import sys
import re
import unittest
from unittest.mock import Mock, patch
from unittest.mock import patch, MagicMock, mock_open
import openai
sys.path.append(f"{pathlib.Path(__file__).parent.resolve()}/../src")
from pyrizz import pyrizz

class Tests:

    # Tests if this returns a list
    def test_get_lines_list(self):
        actual = pyrizz.get_lines('all')
        assert isinstance(actual, list)
    
    # Tests if this returns an empty or not a proper category
    def test_get_lines_nonempty(self):
        actual = pyrizz.get_lines()
        assert isinstance(actual, list)
        assert bool(actual) is True


    # Tests if this returns a string
    def test_get_random_line_str(self):
        actual = pyrizz.get_random_line()
        assert isinstance(actual, str)

    # Tests if the get random line is a non-empty value
    def test_get_random_line_nonempty(self):
        actual = pyrizz.get_random_line()
        assert isinstance(actual, str)
        assert len(actual) > 0

    # Tests if the get random line is a empty value
    def test_get_random_line_empty(self):
        actual = pyrizz.get_random_line()
        assert isinstance(actual, str)
        assert bool(actual)

    # Tests if the get random line is long enough
    def test_get_random_line_long_enough(self):
        actual = pyrizz.get_random_line()
        assert isinstance(actual, str)
        assert len(actual) > 1


    # Tests if this returns a string
    def test_get_random_category_line_str(self):
        actual = pyrizz.get_random_category_line("romantic")
        assert isinstance(actual, str)

    # Tests if the get random category line is a non-empty value
    def test_get_random_category_line_nonempty(self):
        actual = pyrizz.get_random_category_line("romantic")
        assert isinstance(actual, str)
        assert len(actual) > 0

    # Tests if the get random cateogory line is a empty value
    def test_get_random_category_line_empty(self):
        actual = pyrizz.get_random_category_line()
        assert isinstance(actual, str)
        assert bool(actual)

    # Tests if the get random line is long enough
    def test_get_random_category_line_longenough(self):
        actual = pyrizz.get_random_category_line("romantic")
        assert isinstance(actual, str)
        assert len(actual) > 0
    
    # Note: Not testing init_openai(key) because there is not any logic; just working on openai API functionality, authentication, and connection."""
    # Tests if the input for ai line is empty
    def test_get_ai_line_empty(self):
        helper = Helper
        mock_client = helper.create_get_mock()
        actual = pyrizz.get_ai_line("", mock_client)
        expected = "Please specify a category."
        assert actual.strip() == expected.strip()

    # Tests if the input is way too long
    def test_get_ai_line_long(self):
        helper = Helper
        mock_client = helper.create_get_mock()
        actual = pyrizz.get_ai_line("This is a very long category that is definitely more than 50 characters long.", mock_client)
        expected = "Please specify a category that is less than 50 characters."
        assert actual.strip() == expected.strip()

    # Tests if the input for ai line actually results in a string
    def test_get_ai_line_str(self):
        helper = Helper
        mock_client = helper.create_get_mock()
        actual = pyrizz.get_ai_line("test", mock_client)
        assert isinstance(actual, str)

    # Tests if the rate line is empty
    def test_rate_line_empty(self):
        helper = Helper
        mock_client = helper.create_rate_mock()
        actual = pyrizz.rate_line("", mock_client)  
        assert actual == "No pickup line? You gotta use our other features before you come here buddy."

    # Tests if the rate line function follows a specific format
    def test_rate_line_format(self):
        helper = Helper
        mock_client = helper.create_rate_mock()
        actual = pyrizz.rate_line("Do you come with Wi-Fi? Because I'm really feeling a connection.", mock_client)
        assert re.match(r'\d+/10 - .+', actual) is not None

    #Tests if the rate line function returns 
    def test_rate_line_gibberish(self):
        helper = Helper
        mock_client = helper.create_rate_mock()
        actual = pyrizz.rate_line("jwrkf", mock_client)
        assert re.match(r'.+', actual) is not None
     
    # Tests for user input validation 
    def test_is_line_valid_length(self):
        long_line = "x" * 141  
        assert not pyrizz.is_line_valid(long_line), "Expected the line to be flagged as too long."
    
    # Test for invalid tempalte number
    def test_create_line_invalid_template_number(self):
        _, message = pyrizz.create_line(999, ["word1", "word2"])
        assert message == "Template number out of range. Please choose between 0 and {}.".format(len(pyrizz.templates) - 1)
    
    # Test for incorrect word count 
    def test_create_line_incorrect_word_count(self):
        templates = ["Template with one placeholder: {}"]
        with patch('pyrizz.templates', new=templates):
            _, message = pyrizz.create_line(0, ["word1", "word2"])
            expected_message = "Incorrect number of words provided for the placeholders. Expected 1, got 2."
            assert message == expected_message
    
    # Test for non-integer template number
    @patch('builtins.input', side_effect=["not_an_integer"])
    def test_get_user_input_for_line_noninteger(self, mock_input):
        with patch('pyrizz.templates', new=["Some template"]):
            template_number, words = pyrizz.get_user_input_for_line()
            assert template_number is None and words is None
    
    # Test for out of range template numebr
    @patch('builtins.input', side_effect=["99", "word1, word2"])
    def test_get_user_input_for_line_out_of_range(self, mock_input):
        templates = ["Template 0"]
        with patch('pyrizz.templates', new=templates):
            template_number, words = pyrizz.get_user_input_for_line()
            assert template_number is None and words is None
    
    # Test for incorrect number of words provided
    @patch('builtins.input', side_effect=["0", "word1, word2, word3"])
    def test_get_user_input_for_line_incorrect_word_count(self, mock_input):
        templates = ["Template with two placeholders: {}, {}"]
        with patch('pyrizz.templates', new=templates):
            template_number, words = pyrizz.get_user_input_for_line()
            assert template_number is None and words is None


class Helper:
    def create_get_mock():
        mock_client = MagicMock()

        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "Test content"
                    }
                }
            ]
        }
        mock_client.ChatCompletion.create.return_value = MagicMock(**mock_response)

        return mock_client
    
    def create_rate_mock():
        mock_client = MagicMock()

        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "10/10 - Short, sweet, and guaranteed to make anyone blush!"
                    }
                }
            ]
        }
        mock_client.ChatCompletion.create.return_value = MagicMock(**mock_response)

        return mock_client
