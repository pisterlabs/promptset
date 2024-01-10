""" Unit tests for the gpt module. """
import unittest
from unittest.mock import Mock, patch

from gpt import generate_completion


class TestGPT(unittest.TestCase):
    """Test the gpt module."""

    @patch("gpt.openai.ChatCompletion.create")
    def test_generate_completion(self, mock_create):
        """Test that the completion is generated correctly."""
        # Mock the returned object from openai.ChatCompletion.create
        mock_completion = Mock()
        mock_completion.choices = [Mock(message={"content": "Generated text"})]
        mock_create.return_value = mock_completion

        # Call the function to test
        result = generate_completion("gpt-3.5-turbo", "Role", "Prompt")

        # Assert that openai.ChatCompletion.create is called with the expected arguments
        mock_create.assert_called_with(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Role"},
                {"role": "user", "content": "Prompt"},
            ],
        )

        # Assert that the function returns the expected result
        self.assertEqual(result, "Generated text")


if __name__ == "__main__":
    unittest.main()
