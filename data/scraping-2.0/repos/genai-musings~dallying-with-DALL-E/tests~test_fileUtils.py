"""fileUtils class tests."""
import unittest
from unittest.mock import patch, Mock
import os
import sys
import openai

# Add the root folder to the Python module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the DALL·E Image Generator class from dalleImage.py
from dalleImage import dalleImage

# A test suite for the DALL·E Image Generator class.
class TestDalleImage(unittest.TestCase):
    def setUp(self):
        """
        Initialize test setup.

        This method is called before each test case and sets up the DALL·E Image Generator
        instance with a mock API key for testing.
        """
        self.api_key = 'api_key'
        self.dalle = dalleImage(self.api_key)

    def test_generate_image(self):
        """
        Test the generate_image method of the DALL·E Image Generator.

        This test case focuses on the generate_image method and uses mocking techniques.
        The actual OpenAI API call is replaced with a mock to avoid using real credentials.

        It checks if the generate_image method correctly invokes the OpenAI API, and if it
        returns the expected response.
        """
        prompt = "Generate an image"
        n = 2
        size = "256x256"
        expected_response = {"image": "generated_image.png"}

        # Mock the OpenAI API call
        mock_image_create = Mock(return_value=expected_response)
        with patch.object(openai.Image, "create", mock_image_create):
            # Call the generate_image method
            response = self.dalle.generate_image(prompt, n, size)

        # Check if the API call was made with the correct arguments
        mock_image_create.assert_called_once_with(prompt=prompt, n=n, size=size)

        # Check if the response matches the expected response
        self.assertEqual(response, expected_response)

if __name__ == '__main__':
    unittest.main()
