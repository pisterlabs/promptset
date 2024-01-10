import unittest
from helper.main import Helper
from my_helper import cli, check_api_key
import openai

class HelperTestCase(unittest.TestCase):
    """tests for helper"""

    def test_fetch_examples(self):
        """this test tests the data type returned by fetch_examples"""
        data = Helper().fetch_examples()
        self.assertEqual(type(data), list)
        print("passed 1")

    def test_make_prompt(self):
        query = Helper().make_prompt()
        self.assertEqual(type(query), str)
        print("passed 2")

    def test_no_api_key(self):
        openai.api_key = None
        check_api_key()
        self.assertEqual(check_api_key(), 'API_KEY is None')
        print('passed 3')


if __name__ == '__main__':
    unittest.main()
